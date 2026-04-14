import os, sys, time, struct, mmap, resource, binascii, hashlib
import multiprocessing as mp

# --- 1. HYBRID CONFIGURATION (Fixed to match your local script) ---
IS_HPC = os.path.exists("/scratch/cse/phd/csz258233")

if IS_HPC:
    # HPC PATHS
    env_site_packages = "/scratch/cse/phd/csz258233/py39_env/lib/python3.9/site-packages"
    PROJECT_ROOT = "/scratch/cse/phd/csz258233/col7560/et-bert"
    SOURCE_PCAP_DIR = os.path.join(PROJECT_ROOT, "pcaps")
    NUM_WRITERS = 28 # Your HPC Task count
else:
    # LOCAL PATHS (Exactly like your local script)
    env_site_packages = None 
    PROJECT_ROOT = os.getcwd() 
    SOURCE_PCAP_DIR = os.path.join(PROJECT_ROOT, "pcap")
    NUM_WRITERS = 4

# Add custom environment to path if on HPC
if IS_HPC and env_site_packages and env_site_packages not in sys.path:
    sys.path.insert(0, env_site_packages)

CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpora")
SHARD_DIR = os.path.join(CORPUS_DIR, "shards")
WORD_NAME = "encrypted_burst.txt"

# Ring Buffer Config (Your friend's logic)
RING_SIZE = 256 * 1024 * 1024
HEADER_SIZE, WRITE_POS_OFF, READ_POS_OFF, DONE_OFF = 64, 0, 8, 16
_rec, _u64, _u32 = struct.Struct("<IIII"), struct.Struct("<Q"), struct.Struct("<I")

# --- 2. YOUR EXTRACTION LOGIC ---

def bigram_generation(packet_hex, packet_len=64):
    """Your Bigram logic: 'a1 b2 c3 '"""
    # Using your 'cut' logic logic implicitly here for speed
    truncated = packet_hex[:2*packet_len]
    return " ".join([truncated[i:i+2] for i in range(0, len(truncated)-1, 2)])

def extract_packet_info(ts, raw):
    """Zero-copy header slice for session ID"""
    try:
        # Standard IPv4 offsets
        src_ip = ".".join(map(str, raw[26:30]))
        dst_ip = ".".join(map(str, raw[30:34]))
        ips = sorted([src_ip, dst_ip])
        key = f"{ips[0]}_{ips[1]}" 
        direction = 1 if src_ip == ips[0] else -1
        return key, (ts, direction, binascii.hexlify(raw).decode())
    except:
        return None, None

# --- 3. SHARED MEMORY RING BUFFER UTILS ---

def _ring_write(mm, frame):
    flen = len(frame)
    while True:
        wp, rp = _u64.unpack_from(mm, WRITE_POS_OFF)[0], _u64.unpack_from(mm, READ_POS_OFF)[0]
        if (RING_SIZE - ((wp - rp) % RING_SIZE) - 1) >= (4 + flen): break
        time.sleep(0.00005)
    off = HEADER_SIZE + (wp % RING_SIZE)
    _u32.pack_into(mm, off, flen)
    wp_new = wp + 4
    mm[HEADER_SIZE + (wp_new % RING_SIZE):HEADER_SIZE + (wp_new % RING_SIZE) + flen] = frame
    _u64.pack_into(mm, WRITE_POS_OFF, wp_new + flen)

def _ring_read_all(mm):
    while True:
        wp, rp = _u64.unpack_from(mm, WRITE_POS_OFF)[0], _u64.unpack_from(mm, READ_POS_OFF)[0]
        if wp == rp:
            if _u64.unpack_from(mm, DONE_OFF)[0]: return
            time.sleep(0.00005); continue
        off = HEADER_SIZE + (rp % RING_SIZE)
        flen = _u32.unpack_from(mm, off)[0]
        rp_new = rp + 4
        frame = mm[HEADER_SIZE + (rp_new % RING_SIZE):HEADER_SIZE + (rp_new % RING_SIZE) + flen]
        _u64.pack_into(mm, READ_POS_OFF, rp_new + flen)
        yield frame

# --- 4. THE WRITER (Contiguous Session Aggregator) ---

def writer_worker(worker_id, mm):
    os.makedirs(SHARD_DIR, exist_ok=True)
    shard_path = os.path.join(SHARD_DIR, f"shard_{worker_id}.txt")
    
    # Store all packets for a session in a dict: key -> [(ts, dir, hex), ...]
    session_packets = {} 

    print(f"[Writer {worker_id}] Collecting sessions...")
    for frame in _ring_read_all(mm):
        nul = frame.index(b'\x00')
        key = frame[:nul].decode()
        data = eval(frame[nul+1:].decode()) # (ts, dir, hex)
        
        if key not in session_packets:
            session_packets[key] = []
        session_packets[key].append(data)

    print(f"[Writer {worker_id}] Writing contiguous blocks to {shard_path}")
    with open(shard_path, "w") as f:
        for sid in session_packets:
            # Sort packets within this session by time
            packets = sorted(session_packets[sid], key=lambda x: x[0])
            
            # Extract bursts for this specific session
            current_burst_hex = ""
            last_dir = packets[0][1]
            for ts, dr, hex_val in packets:
                if dr != last_dir:
                    f.write(bigram_generation(current_burst_hex) + "\n")
                    current_burst_hex = hex_val
                    last_dir = dr
                else:
                    current_burst_hex += hex_val
            
            # Final burst and session separator
            f.write(bigram_generation(current_burst_hex) + "\n\n")
            session_packets[sid] = None # Help Garbage Collector

# --- 5. MAIN ---

def reader_logic(mmaps, done_event):
    # Find the PCAP in your local/HPC directory
    pcap_files = [f for f in os.listdir(SOURCE_PCAP_DIR) if f.lower().endswith(('.pcap', '.pcapng'))]
    if not pcap_files:
        print("CRITICAL: No PCAP found in", SOURCE_PCAP_DIR)
        done_event.set(); return
    
    target_pcap = os.path.join(SOURCE_PCAP_DIR, pcap_files[0])
    print(f"[Reader] Processing: {target_pcap}")

    with open(target_pcap, "rb") as f:
        f.seek(24) # Skip PCAP global header
        while True:
            hdr = f.read(16)
            if len(hdr) < 16: break
            ts_s, ts_us, cap_len, _ = _rec.unpack(hdr)
            raw = f.read(cap_len)
            key, data = extract_packet_info(ts_s + ts_us*1e-6, raw)
            if key:
                frame = key.encode() + b'\x00' + str(data).encode()
                # Hash routing ensures session packets stay together in one ring
                _ring_write(mmaps[hash(key) % len(mmaps)], frame)
    
    done_event.set()

if __name__ == "__main__":
    # Maximize limits
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except: pass

    mmaps = [mmap.mmap(-1, HEADER_SIZE + RING_SIZE) for _ in range(NUM_WRITERS)]
    for mm in mmaps: mm[:HEADER_SIZE] = b'\x00' * HEADER_SIZE
    
    done_ev = mp.Event()
    
    # Launch Workers
    writers = [mp.Process(target=writer_worker, args=(i, mmaps[i])) for i in range(NUM_WRITERS)]
    for p in writers: p.start()
    
    # Launch Reader
    reader_proc = mp.Process(target=reader_logic, args=(mmaps, done_ev))
    reader_proc.start()
    
    reader_proc.join()
    # Signal writers to finish
    for mm in mmaps: _u64.pack_into(mm, DONE_OFF, 1)
    for p in writers: p.join()
    
    print(f"Extraction complete. Shards in {SHARD_DIR}")
    print(f"Final Merge: cat {SHARD_DIR}/shard_*.txt > {CORPUS_DIR}/{WORD_NAME}")