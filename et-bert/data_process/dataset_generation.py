import os, sys, time, struct, mmap, resource, binascii, hashlib
import multiprocessing as mp

# --- 1. HYBRID CONFIGURATION ---
IS_HPC = os.path.exists("/scratch/cse/phd/csz258233")

if IS_HPC:
    PROJECT_ROOT = "/scratch/cse/phd/csz258233/col7560/et-bert"
    SOURCE_PCAP_DIR = os.path.join(PROJECT_ROOT, "pcaps")
    NUM_WRITERS = 28 # Matched to TSK on HPC
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
    """Generates ET-BERT bigrams (sliding window of 2 bytes)"""
    res = []
    # Step by 2 (byte alignment), take 4 (sliding window)
    for i in range(0, len(packet_hex) - 2, 2):
        token = packet_hex[i:i+4]
        if len(token) == 4:
            res.append(token)
        if len(res) >= packet_len:
            break
    return " ".join(res)

def extract_packet_info(ts, raw):
    """Fast header slice for session ID"""
    try:
        # Standard IPv4 offset (Eth=14, IPs at 26-34)
        src_ip = ".".join(map(str, raw[26:30]))
        dst_ip = ".".join(map(str, raw[30:34]))
        ips = sorted([src_ip, dst_ip])
        key = f"{ips[0]}_{ips[1]}" 
        direction = 1 if src_ip == ips[0] else -1
        return key, ts, direction, binascii.hexlify(raw).decode()
    except:
        return None

# --- 3. SYMMETRIC RING BUFFER UTILS ---

def _ring_write(mm, frame):
    """Writes frame byte-by-byte to ensure wrap-around never crashes"""
    flen = len(frame)
    total_need = 4 + flen
    while True:
        wp = _u64.unpack_from(mm, WRITE_POS_OFF)[0]
        rp = _u64.unpack_from(mm, READ_POS_OFF)[0]
        if (RING_SIZE - ((wp - rp) % RING_SIZE) - 1) >= total_need: break
        time.sleep(0.0001)

    # Write 4-byte length byte-by-byte
    header_bytes = _u32.pack(flen)
    for i in range(4):
        mm[HEADER_SIZE + ((wp + i) % RING_SIZE)] = header_bytes[i]
    
    # Write Data byte-by-byte
    wp_data = wp + 4
    for i in range(flen):
        mm[HEADER_SIZE + ((wp_data + i) % RING_SIZE)] = frame[i]

    _u64.pack_into(mm, WRITE_POS_OFF, wp_data + flen)

def _ring_read_all(mm):
    """Reads frame byte-by-byte to stay in sync with Reader"""
    while True:
        wp = _u64.unpack_from(mm, WRITE_POS_OFF)[0]
        rp = _u64.unpack_from(mm, READ_POS_OFF)[0]
        if wp == rp:
            if _u64.unpack_from(mm, DONE_OFF)[0]: return
            time.sleep(0.0001); continue

        # Read 4-byte length byte-by-byte
        l_bytes = bytearray(4)
        for i in range(4):
            l_bytes[i] = mm[HEADER_SIZE + ((rp + i) % RING_SIZE)]
        flen = _u32.unpack(l_bytes)[0]
        
        # Read Data byte-by-byte
        rp_data = rp + 4
        frame = bytearray(flen)
        for i in range(flen):
            frame[i] = mm[HEADER_SIZE + ((rp_data + i) % RING_SIZE)]
            
        _u64.pack_into(mm, READ_POS_OFF, rp_data + flen)
        yield bytes(frame)

# --- 4. THE WRITER (Saves Session Groups) ---

def writer_worker(worker_id, mm):
    os.makedirs(SHARD_DIR, exist_ok=True)
    shard_path = os.path.join(SHARD_DIR, f"shard_{worker_id}.txt")
    session_packets = {} 
    pkt_count = 0

    with open(shard_path, "w") as f_out:
        print(f"[Writer {worker_id}] Processing...")
        for frame in _ring_read_all(mm):
            try:
                # Format: key|ts|dir|hex
                parts = frame.decode(errors='ignore').split('|')
                if len(parts) < 4: continue
                
                key, ts, dr, hx = parts[0], float(parts[1]), int(parts[2]), parts[3]
                
                if key not in session_packets:
                    session_packets[key] = []
                session_packets[key].append((ts, dr, hx))
                pkt_count += 1

                # Flush inactive sessions every 500k packets to keep RAM clean
                if pkt_count % 500000 == 0:
                    curr_ts = ts
                    expired = [sid for sid, pks in session_packets.items() if (curr_ts - pks[-1][0]) > IDLE_TIMEOUT]
                    for sid in expired:
                        write_session_to_file(f_out, session_packets[sid])
                        del session_packets[sid]
            except: continue

        # Final flush
        for sid in session_packets:
            write_session_to_file(f_out, session_packets[sid])
    
    print(f"[Writer {worker_id}] Completed.")

def write_session_to_file(f_handle, packets):
    if not packets: return
    packets.sort(key=lambda x: x[0]) # GROUPED and ORDERED
    
    curr_burst = ""
    last_dir = packets[0][1]
    
    for _, dr, hex_val in packets:
        if dr != last_dir:
            f_handle.write(bigram_generation(curr_burst) + "\n")
            curr_burst = hex_val
            last_dir = dr
        else:
            curr_burst += hex_val
    
    f_handle.write(bigram_generation(curr_burst) + "\n\n")

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
            res = extract_packet_info(ts_s + ts_us*1e-6, raw)
            if res:
                key, ts, dr, hx = res
                # Pipe-delimited string is 10x faster than eval()
                frame_str = f"{key}|{ts}|{dr}|{hx}"
                _ring_write(mmaps[hash(key) % len(mmaps)], frame_str.encode())
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
    print("FINISHED.")
