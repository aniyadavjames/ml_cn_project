import os, sys, time, struct, mmap, resource, binascii, hashlib, ast
import multiprocessing as mp

# --- 1. HYBRID CONFIGURATION ---
IS_HPC = os.path.exists("/scratch/cse/phd/csz258233")

if IS_HPC:
    PROJECT_ROOT = "/scratch/cse/phd/csz258233/col7560/et-bert"
    SOURCE_PCAP_DIR = os.path.join(PROJECT_ROOT, "pcaps")
    NUM_WRITERS = 28 
else:
    PROJECT_ROOT = os.getcwd() 
    SOURCE_PCAP_DIR = os.path.join(PROJECT_ROOT, "pcap")
    NUM_WRITERS = 4

CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpora")
SHARD_DIR = os.path.join(CORPUS_DIR, "shards")
WORD_NAME = "encrypted_burst.txt"

# Ring Buffer Settings
RING_SIZE = 256 * 1024 * 1024
HEADER_SIZE, WRITE_POS_OFF, READ_POS_OFF, DONE_OFF = 64, 0, 8, 16
_rec, _u64, _u32 = struct.Struct("<IIII"), struct.Struct("<Q"), struct.Struct("<I")
IDLE_TIMEOUT = 60.0

# --- 2. EXTRACTION LOGIC ---

def bigram_generation(packet_hex, packet_len=64):
    truncated = packet_hex[:2*packet_len]
    return " ".join([truncated[i:i+2] for i in range(0, len(truncated)-1, 2)])

def extract_packet_info(ts, raw):
    try:
        # Fast slice for IPv4 (Standard Ethernet)
        src_ip = ".".join(map(str, raw[26:30]))
        dst_ip = ".".join(map(str, raw[30:34]))
        ips = sorted([src_ip, dst_ip])
        key = f"{ips[0]}_{ips[1]}" 
        direction = 1 if src_ip == ips[0] else -1
        return key, (ts, direction, binascii.hexlify(raw).decode())
    except:
        return None, None

# --- 3. RING BUFFER UTILS ---

def _ring_write(mm, frame):
    flen = len(frame)
    total_need = 4 + flen

    while True:
        wp = _u64.unpack_from(mm, WRITE_POS_OFF)[0]
        rp = _u64.unpack_from(mm, READ_POS_OFF)[0]
        free = RING_SIZE - ((wp - rp) % RING_SIZE) - 1
        if free >= total_need: break
        time.sleep(0.00005)

    # 1. Write the 4-byte length header (handles wrap-around)
    off = HEADER_SIZE + (wp % RING_SIZE)
    if (wp % RING_SIZE) + 4 <= RING_SIZE:
        _u32.pack_into(mm, off, flen)
    else:
        # Header itself is split (rare but possible)
        header_bytes = _u32.pack(flen)
        for i in range(4):
            mm[HEADER_SIZE + ((wp + i) % RING_SIZE)] = header_bytes[i]
    
    wp_new = wp + 4

    # 2. Write the frame bytes (The "Slice" Fix)
    curr_off = wp_new % RING_SIZE
    if curr_off + flen <= RING_SIZE:
        # Fits in one piece
        mm[HEADER_SIZE + curr_off : HEADER_SIZE + curr_off + flen] = frame
    else:
        # MUST SPLIT: Write part to end, part to beginning
        end_chunk_size = RING_SIZE - curr_off
        mm[HEADER_SIZE + curr_off : HEADER_SIZE + RING_SIZE] = frame[:end_chunk_size]
        mm[HEADER_SIZE : HEADER_SIZE + (flen - end_chunk_size)] = frame[end_chunk_size:]

    _u64.pack_into(mm, WRITE_POS_OFF, wp_new + flen)
    
def _ring_read_all(mm):
    while True:
        wp, rp = _u64.unpack_from(mm, WRITE_POS_OFF)[0], _u64.unpack_from(mm, READ_POS_OFF)[0]
        if wp == rp:
            if _u64.unpack_from(mm, DONE_OFF)[0]: return
            time.sleep(0.0001); continue
        off = HEADER_SIZE + (rp % RING_SIZE)
        flen = _u32.unpack_from(mm, off)[0]
        rp_new = rp + 4
        # Handle wrap-around frame extraction
        frame = bytearray(flen)
        for i in range(flen):
            frame[i] = mm[HEADER_SIZE + ((rp_new + i) % RING_SIZE)]
        _u64.pack_into(mm, READ_POS_OFF, rp_new + flen)
        yield bytes(frame)

# --- 4. THE WRITER (With Flush Logic) ---

def writer_worker(worker_id, mm):
    os.makedirs(SHARD_DIR, exist_ok=True)
    shard_path = os.path.join(SHARD_DIR, f"shard_{worker_id}.txt")
    
    # key -> list of packets
    session_packets = {} 
    pkt_count = 0

    with open(shard_path, "w") as f_out:
        print(f"[Writer {worker_id}] Processing...")
        for frame in _ring_read_all(mm):
            try:
                nul = frame.index(b'\x00')
                key = frame[:nul].decode(errors='ignore')
                data = ast.literal_eval(frame[nul+1:].decode(errors='ignore'))
                
                if key not in session_packets:
                    session_packets[key] = []
                session_packets[key].append(data)
                pkt_count += 1

                # --- AUTO-FLUSH TO PROTECT RAM (Every 1M packets) ---
                if pkt_count % 1000000 == 0:
                    current_ts = data[0]
                    expired = [sid for sid, pks in session_packets.items() if (current_ts - pks[-1][0]) > IDLE_TIMEOUT]
                    for sid in expired:
                        write_session_to_file(f_out, session_packets[sid])
                        del session_packets[sid]
            except: continue

        # Final Flush at end of file
        for sid in session_packets:
            write_session_to_file(f_out, session_packets[sid])
    
    print(f"[Writer {worker_id}] Completed.")

def write_session_to_file(f_handle, packets):
    """Sorts and writes a single session block"""
    if not packets: return
    packets.sort(key=lambda x: x[0])
    
    current_burst_hex = ""
    last_dir = packets[0][1]
    
    for ts, dr, hex_val in packets:
        if dr != last_dir:
            f_handle.write(bigram_generation(current_burst_hex) + "\n")
            current_burst_hex = hex_val
            last_dir = dr
        else:
            current_burst_hex += hex_val
    
    f_handle.write(bigram_generation(current_burst_hex) + "\n\n")

# --- 5. MAIN ---

def reader_logic(mmaps, done_event):
    pcap_files = [f for f in os.listdir(SOURCE_PCAP_DIR) if f.lower().endswith(('.pcap', '.pcapng'))]
    if not pcap_files: return
    
    target_pcap = os.path.join(SOURCE_PCAP_DIR, pcap_files[0])
    print(f"[Reader] Streaming: {target_pcap}")

    with open(target_pcap, "rb") as f:
        f.seek(24) 
        while True:
            hdr = f.read(16)
            if len(hdr) < 16: break
            ts_s, ts_us, cap_len, _ = _rec.unpack(hdr)
            raw = f.read(cap_len)
            key, data = extract_packet_info(ts_s + ts_us*1e-6, raw)
            if key:
                frame = key.encode() + b'\x00' + str(data).encode()
                _ring_write(mmaps[hash(key) % len(mmaps)], frame)
    done_event.set()

if __name__ == "__main__":
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except: pass

    mmaps = [mmap.mmap(-1, HEADER_SIZE + RING_SIZE) for _ in range(NUM_WRITERS)]
    for mm in mmaps: mm[:HEADER_SIZE] = b'\x00' * HEADER_SIZE
    
    done_ev = mp.Event()
    writers = [mp.Process(target=writer_worker, args=(i, mmaps[i])) for i in range(NUM_WRITERS)]
    for p in writers: p.start()
    
    reader_proc = mp.Process(target=reader_logic, args=(mmaps, done_ev))
    reader_proc.start()
    
    reader_proc.join()
    for mm in mmaps: _u64.pack_into(mm, DONE_OFF, 1)
    for p in writers: p.join()
    print("FINISHED. Check shards directory.")