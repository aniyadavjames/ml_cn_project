"""
Root cause of 29K/s: mp.Queue pickles every (key, record) through a pipe.
Fix: shared mmap ring buffers — zero pickle, zero pipe, zero copy.

Architecture:
  One ring buffer per writer (6 total), each 256 MB of anonymous shared memory.
  Reader writes frames directly into the mmap.
  Writer reads frames directly from the mmap.
  No pickle. No pipe. No Queue thread.

Frame format inside ring:
  [4-byte frame_len][key_bytes][0x00][record_bytes]

Ring header (64 bytes at start of each mmap):
  offset 0:  write_pos (uint64) — advanced by reader
  offset 8:  read_pos  (uint64) — advanced by writer
  offset 16: done      (uint64) — set to 1 when reader finishes
"""

import os, sys, time, struct, mmap, resource
import multiprocessing as mp

try:
    import dpkt
    USE_DPKT = True
except ImportError:
    USE_DPKT = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_PCAP     = "scratch/flow/all/all_packets_first_100.pcap"
OUTPUT_DIR     = "scratch/flows_split/all3"

NUM_WRITERS    = 6
NUM_READERS    = 1
RAM_LIMIT_PCT  = 88
PROGRESS_EVERY = 2_000_000
DISK_CHUNK     = 4 * 1024 * 1024     # 4 MB write chunks in phase 2

# 6 × 256 MB = 1.5 GB total shared memory — negligible vs 400 GB
RING_SIZE      = 256 * 1024 * 1024

HEADER_SIZE   = 64
WRITE_POS_OFF = 0
READ_POS_OFF  = 8
DONE_OFF      = 16

_rec  = struct.Struct("<IIII")
_u64  = struct.Struct("<Q")
_u32  = struct.Struct("<I")

PCAP_HDR = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)

# ── pcap ──────────────────────────────────────────────────────────────────────
def make_record(ts, raw):
    s  = int(ts)
    us = int(round((ts - s) * 1e6))
    n  = len(raw)
    return _rec.pack(s, us, n, n) + raw

# ── extraction ────────────────────────────────────────────────────────────────
_ntoa_fn = None
def _ntoa(b):
    global _ntoa_fn
    if _ntoa_fn is None:
        import socket; _ntoa_fn = socket.inet_ntoa
    return _ntoa_fn(b)

def extract_dpkt(ts, raw):
    try:
        eth = dpkt.ethernet.Ethernet(raw)
        ip  = eth.data
        if not isinstance(ip, dpkt.ip.IP): return None, None
        seg = ip.data
        if   isinstance(seg, dpkt.tcp.TCP): sp, dp = seg.sport, seg.dport
        elif isinstance(seg, dpkt.udp.UDP): sp, dp = seg.sport, seg.dport
        else:                               sp, dp = 0, 0
        # key as flat string — cheap encode/decode, no tuple pickling
        key = f"{_ntoa(ip.src)},{_ntoa(ip.dst)},{sp},{dp},{ip.p}"
        return key, make_record(ts, raw)
    except Exception:
        return None, None

def extract_scapy(pkt):
    from scapy.all import IP, TCP, UDP
    if IP not in pkt: return None, None
    ip = pkt[IP]
    if   TCP in pkt: sp, dp = pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt: sp, dp = pkt[UDP].sport, pkt[UDP].dport
    else:            sp, dp = 0, 0
    key = f"{ip.src},{ip.dst},{sp},{dp},{ip.proto}"
    return key, make_record(float(pkt.time), bytes(pkt))

# ── ring buffer ───────────────────────────────────────────────────────────────
def _ring_write(mm, frame: bytes):
    """Write one frame, spin-waiting if ring is full."""
    flen       = len(frame)
    total_need = 4 + flen

    while True:
        wp   = _u64.unpack_from(mm, WRITE_POS_OFF)[0]
        rp   = _u64.unpack_from(mm, READ_POS_OFF)[0]
        free = RING_SIZE - ((wp - rp) % RING_SIZE) - 1
        if free >= total_need: break
        time.sleep(0.00005)   # ring full — writer is lagging

    # write 4-byte length
    off = HEADER_SIZE + (wp % RING_SIZE)
    _u32.pack_into(mm, off, flen)
    wp_new = wp + 4

    # write frame bytes with wrap-around
    src = 0
    rem = flen
    while rem:
        off   = HEADER_SIZE + (wp_new % RING_SIZE)
        chunk = min(rem, RING_SIZE - (wp_new % RING_SIZE))
        mm[off:off+chunk] = frame[src:src+chunk]
        wp_new += chunk
        src    += chunk
        rem    -= chunk

    _u64.pack_into(mm, WRITE_POS_OFF, wp_new)

def _ring_read_all(mm):
    """Generator yielding frames until ring is done+empty."""
    while True:
        wp = _u64.unpack_from(mm, WRITE_POS_OFF)[0]
        rp = _u64.unpack_from(mm, READ_POS_OFF)[0]
        if wp == rp:
            if _u64.unpack_from(mm, DONE_OFF)[0]:
                return
            time.sleep(0.00005)
            continue

        # read length
        off  = HEADER_SIZE + (rp % RING_SIZE)
        flen = _u32.unpack_from(mm, off)[0]
        rp_new = rp + 4

        # read frame bytes with wrap-around
        frame = bytearray(flen)
        dst   = 0
        rem   = flen
        while rem:
            off   = HEADER_SIZE + (rp_new % RING_SIZE)
            chunk = min(rem, RING_SIZE - (rp_new % RING_SIZE))
            frame[dst:dst+chunk] = mm[off:off+chunk]
            rp_new += chunk
            dst    += chunk
            rem    -= chunk

        _u64.pack_into(mm, READ_POS_OFF, rp_new)
        yield bytes(frame)

# ── RAM watchdog ──────────────────────────────────────────────────────────────
def ram_watchdog(pause_event, stop_event):
    if not HAS_PSUTIL:
        stop_event.wait(); return
    while not stop_event.is_set():
        pct = psutil.virtual_memory().percent
        if   pct >= RAM_LIMIT_PCT    and not pause_event.is_set():
            print(f"[RAM] {pct:.0f}% — pausing", flush=True); pause_event.set()
        elif pct < RAM_LIMIT_PCT - 5 and     pause_event.is_set():
            print(f"[RAM] {pct:.0f}% — resuming", flush=True); pause_event.clear()
        time.sleep(1)

# ── done watcher ──────────────────────────────────────────────────────────────
def done_watcher(done_flags, mmaps):
    for f in done_flags: f.wait()
    for mm in mmaps:
        _u64.pack_into(mm, DONE_OFF, 1)
    print("[Watcher] all readers done", flush=True)

# ── reader ────────────────────────────────────────────────────────────────────
def reader(reader_id, byte_start, byte_end, mmaps, pause_event, done_flags):
    n    = len(mmaps)
    sent = skipped = 0
    t0   = time.time()

    def route(key, record):
        frame = key.encode() + b'\x00' + record
        _ring_write(mmaps[hash(key) % n], frame)

    if not USE_DPKT:
        if reader_id == 0:
            from scapy.all import PcapReader
            print("[Reader] scapy — pip install dpkt for 10x speed", flush=True)
            with PcapReader(INPUT_PCAP) as pr:
                for pkt in pr:
                    while pause_event.is_set(): time.sleep(0.2)
                    key, rec = extract_scapy(pkt)
                    if key is None: skipped += 1; continue
                    route(key, rec)
                    sent += 1
                    if sent % PROGRESS_EVERY == 0:
                        rate = sent / (time.time() - t0)
                        print(f"[Reader] {sent/1e6:.1f}M | {rate/1e3:.0f}K/s", flush=True)
        done_flags[reader_id].set()
        return

    print(f"[Reader {reader_id}] {byte_start/1e9:.2f}–{byte_end/1e9:.2f} GB", flush=True)
    with open(INPUT_PCAP, "rb") as f:
        f.seek(byte_start if byte_start > 0 else 24)
        while True:
            if byte_end > 0 and f.tell() >= byte_end: break
            while pause_event.is_set(): time.sleep(0.2)
            hdr = f.read(16)
            if len(hdr) < 16: break
            ts_s, ts_us, cap_len, _ = _rec.unpack(hdr)
            raw = f.read(cap_len)
            if len(raw) < cap_len: break
            key, rec = extract_dpkt(ts_s + ts_us * 1e-6, raw)
            if key is None: skipped += 1; continue
            route(key, rec)
            sent += 1
            if sent % PROGRESS_EVERY == 0:
                rate = sent / (time.time() - t0)
                eta  = (1_760_000_000 - sent) / rate / 3600
                print(f"[Reader {reader_id}] {sent/1e6:.1f}M | {rate/1e3:.0f}K/s | ETA ~{eta:.1f}h", flush=True)

    done_flags[reader_id].set()
    print(f"[Reader {reader_id}] done — {sent:,} sent, {skipped:,} skipped", flush=True)

# ── writer ────────────────────────────────────────────────────────────────────
def writer_worker(worker_id, mm):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    flow_data  = {}    # key_str -> bytearray
    flow_order = []
    written    = 0
    t0         = time.time()
    

    print(f"[Writer {worker_id}] phase 1 — reading from ring buffer", flush=True)

    for frame in _ring_read_all(mm):
        """try:
            nul = frame.index(b'\x00')
        except ValueError:
            continue  # skip corrupted frame

        try:
            key = frame[:nul].decode(errors='ignore')
        except:
            continue

        record = frame[nul+1:]"""

        nul    = frame.index(b'\x00')
        key    = frame[:nul].decode()
        record = frame[nul+1:]

        if key not in flow_data:
            flow_data[key] = bytearray(PCAP_HDR)
            flow_order.append(key)
        flow_data[key] += record
        written += 1

        if written % PROGRESS_EVERY == 0:
            ram_gb = sum(len(v) for v in flow_data.values()) / 1e9
            rate   = written / (time.time() - t0)
            print(f"[Writer {worker_id}] {written/1e6:.1f}M | "
                  f"{len(flow_data):,} flows | {ram_gb:.2f} GB | {rate/1e3:.0f}K/s", flush=True)

    p1 = time.time() - t0
    total_gb = sum(len(v) for v in flow_data.values()) / 1e9
    print(f"[Writer {worker_id}] phase 1 done {p1:.0f}s — "
          f"{written:,} records | {len(flow_data):,} flows | {total_gb:.2f} GB", flush=True)

    print(f"[Writer {worker_id}] phase 2 — writing {len(flow_order):,} files", flush=True)
    t1 = time.time()

    for i, key in enumerate(flow_order):
        path = os.path.join(OUTPUT_DIR, f"flow_w{worker_id}_{i}.pcap")
        data = flow_data[key]
        mv   = memoryview(data)
        with open(path, "wb") as fh:
            pos = 0
            while pos < len(data):
                end = min(pos + DISK_CHUNK, len(data))
                fh.write(mv[pos:end]); pos = end
        flow_data[key] = None

        if (i + 1) % 20_000 == 0:
            print(f"[Writer {worker_id}] {i+1:,}/{len(flow_order):,} files written", flush=True)

    print(f"[Writer {worker_id}] phase 2 done {time.time()-t1:.0f}s", flush=True)

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_PCAP):
        print(f"ERROR: {INPUT_PCAP} not found"); sys.exit(1)

    file_size = os.path.getsize(INPUT_PCAP)
    n_readers = NUM_READERS if USE_DPKT else 1

    try: soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except: soft = "?"

    print("=" * 62)
    print(f"Input        : {INPUT_PCAP} ({file_size/1e9:.1f} GB)")
    print(f"Output       : {OUTPUT_DIR}")
    print(f"Readers      : {n_readers}   Writers : {NUM_WRITERS}")
    print(f"ulimit -n    : {soft}")
    print(f"Transport    : mmap ring buffers ({NUM_WRITERS}×{RING_SIZE//1024//1024}MB = {NUM_WRITERS*RING_SIZE/1e9:.1f}GB)")
    print(f"Parser       : {'dpkt (fast)' if USE_DPKT else 'scapy'}")
    print("=" * 62)

    # anonymous shared mmap per writer
    mmaps = []
    for _ in range(NUM_WRITERS):
        mm = mmap.mmap(-1, HEADER_SIZE + RING_SIZE)
        mm[:HEADER_SIZE] = b'\x00' * HEADER_SIZE
        mmaps.append(mm)

    pause_ev   = mp.Event()
    stop_ev    = mp.Event()
    done_flags = [mp.Event() for _ in range(n_readers)]

    usable = file_size - 24
    chunk  = usable // n_readers
    ranges = [(0 if i == 0 else 24 + i*chunk,
               file_size if i == n_readers-1 else 24 + (i+1)*chunk)
              for i in range(n_readers)]

    t0 = time.time()

    mp.Process(target=ram_watchdog, args=(pause_ev, stop_ev), daemon=True).start()
    mp.Process(target=done_watcher, args=(done_flags, mmaps),  daemon=True).start()

    writers = [mp.Process(target=writer_worker, args=(i, mmaps[i])) for i in range(NUM_WRITERS)]
    for p in writers: p.start()

    readers = [mp.Process(target=reader, args=(i, ranges[i][0], ranges[i][1],
                                                mmaps, pause_ev, done_flags))
               for i in range(n_readers)]
    for p in readers: p.start()

    for p in readers: p.join()
    for p in writers: p.join()

    elapsed = time.time() - t0
    total_flows = sum(len([f for f in os.listdir(OUTPUT_DIR)
                           if f.startswith(f"flow_w{i}_")])
                      for i in range(NUM_WRITERS))
    print(f"\nDone in {elapsed/3600:.2f}h | flows: {total_flows:,} | out: {OUTPUT_DIR}")