"""
Hard ulimit=1024 solution.

The LRU file-handle cache is useless when unique flows >> handles available.
With 500K flows across 6 writers = ~83K flows/writer but only 136 fds/writer,
99.8% of accesses would evict and reopen — slower than no cache at all.

Real fix: don't keep files open during accumulation at all.
  Phase 1  — writers accumulate records in RAM: dict[key -> bytearray]
             Zero file I/O during this phase. Zero fd usage (beyond queue pipe).
  Phase 2  — once the queue is drained, write each flow's bytearray to disk
             sequentially, one open()+write()+close() per flow.
             Peak concurrent fds during phase 2 = 1 per writer = 6 total.

RAM budget (400 GB):
  189 GB raw packet data  (the data itself)
  ~5–10 GB dict overhead  (bytearray objects + key tuples)
  ~20 GB queue buffers    (6 × 50K × 500B avg)
  ──────────────────────
  ~220 GB peak            fits in 400 GB with 180 GB headroom

Why this is also faster:
  - Phase 1: pure RAM writes (bytearray +=) — no syscalls at all
  - Phase 2: large sequential writes — OS can buffer and flush efficiently
  - No LRU overhead, no reopen() cost, no file object creation in hot loop
"""

import os, sys, time, struct, resource
import multiprocessing as mp
from collections import defaultdict

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
OUTPUT_DIR     = "scratch/flows_split/all2"

NUM_WRITERS    = 6
NUM_READERS    = 2
QUEUE_MAXSIZE  = 500_000
RAM_LIMIT_PCT  = 88
PROGRESS_EVERY = 2_000_000
BATCH_SIZE     = 512

# Phase 2 write buffer: how many bytes to write per fwrite() call
# Larger = fewer syscalls. 4MB is a good match for most storage systems.
DISK_WRITE_CHUNK = 4 * 1024 * 1024   # 4 MB

_rec_struct = struct.Struct("<IIII")

# ── pcap ──────────────────────────────────────────────────────────────────────
PCAP_GLOBAL_HEADER = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)

def make_record(ts: float, raw: bytes) -> bytes:
    s  = int(ts)
    us = int(round((ts - s) * 1_000_000))
    n  = len(raw)
    return _rec_struct.pack(s, us, n, n) + raw

# ── extraction ────────────────────────────────────────────────────────────────
_inet_ntoa = None
def _ntoa(b):
    global _inet_ntoa
    if _inet_ntoa is None:
        import socket; _inet_ntoa = socket.inet_ntoa
    return _inet_ntoa(b)

def extract_dpkt(ts, raw):
    try:
        eth = dpkt.ethernet.Ethernet(raw)
        ip  = eth.data
        if not isinstance(ip, dpkt.ip.IP): return None, None
        seg = ip.data
        if   isinstance(seg, dpkt.tcp.TCP): sp, dp = seg.sport, seg.dport
        elif isinstance(seg, dpkt.udp.UDP): sp, dp = seg.sport, seg.dport
        else:                               sp, dp = 0, 0
        return (_ntoa(ip.src), _ntoa(ip.dst), sp, dp, ip.p), make_record(ts, raw)
    except Exception:
        return None, None

def extract_scapy(pkt):
    from scapy.all import IP, TCP, UDP
    if IP not in pkt: return None, None
    ip = pkt[IP]
    if   TCP in pkt: sp, dp = pkt[TCP].sport, pkt[TCP].dport
    elif UDP in pkt: sp, dp = pkt[UDP].sport, pkt[UDP].dport
    else:            sp, dp = 0, 0
    return (ip.src, ip.dst, sp, dp, ip.proto), make_record(float(pkt.time), bytes(pkt))

# ── RAM watchdog ──────────────────────────────────────────────────────────────
def ram_watchdog(pause_event, stop_event):
    if not HAS_PSUTIL:
        print("[RAM] psutil missing — monitoring disabled", flush=True)
        stop_event.wait(); return
    while not stop_event.is_set():
        vm  = psutil.virtual_memory()
        pct = vm.percent
        if   pct >= RAM_LIMIT_PCT    and not pause_event.is_set():
            print(f"[RAM] {pct:.0f}% ({vm.used/1e9:.1f}/{vm.total/1e9:.1f} GB) — pausing", flush=True)
            pause_event.set()
        elif pct < RAM_LIMIT_PCT - 5 and     pause_event.is_set():
            print(f"[RAM] {pct:.0f}% — resuming", flush=True)
            pause_event.clear()
        time.sleep(1)

# ── reader ────────────────────────────────────────────────────────────────────
def reader(reader_id, byte_start, byte_end, queues, pause_event, done_counter):
    n    = len(queues)
    sent = skipped = 0
    t0   = time.time()

    if not USE_DPKT:
        if reader_id == 0:
            from scapy.all import PcapReader
            print("[Reader] scapy — single reader. pip install dpkt for 2x speed", flush=True)
            with PcapReader(INPUT_PCAP) as pr:
                for pkt in pr:
                    while pause_event.is_set(): time.sleep(0.2)
                    key, rec = extract_scapy(pkt)
                    if key is None: skipped += 1; continue
                    queues[hash(key) % n].put((key, rec))
                    sent += 1
                    if sent % PROGRESS_EVERY == 0:
                        rate = sent / (time.time() - t0)
                        eta  = (1_760_000_000 - sent) / rate / 3600
                        print(f"[Reader] {sent/1e6:.1f}M | {rate/1e3:.0f}K/s | ETA ~{eta:.1f}h", flush=True)
        with done_counter.get_lock(): done_counter.value += 1
        return

    print(f"[Reader {reader_id}] {byte_start/1e9:.2f}–{byte_end/1e9:.2f} GB", flush=True)
    with open(INPUT_PCAP, "rb") as f:
        f.seek(byte_start if byte_start > 0 else 24)
        while True:
            if byte_end > 0 and f.tell() >= byte_end: break
            while pause_event.is_set(): time.sleep(0.2)
            hdr = f.read(16)
            if len(hdr) < 16: break
            ts_s, ts_us, cap_len, _ = _rec_struct.unpack(hdr)
            raw = f.read(cap_len)
            if len(raw) < cap_len: break
            key, rec = extract_dpkt(ts_s + ts_us * 1e-6, raw)
            if key is None: skipped += 1; continue
            queues[hash(key) % n].put((key, rec))
            sent += 1
            if sent % PROGRESS_EVERY == 0:
                rate = sent / (time.time() - t0)
                print(f"[Reader {reader_id}] {sent/1e6:.1f}M | {rate/1e3:.0f}K/s", flush=True)

    with done_counter.get_lock(): done_counter.value += 1
    print(f"[Reader {reader_id}] done — {sent:,} sent, {skipped:,} skipped", flush=True)

# ── writer ────────────────────────────────────────────────────────────────────
def writer_worker(worker_id, queue, total_readers):
    """
    Phase 1: drain queue entirely into RAM (dict[key -> bytearray]).
             No file I/O. No open file handles. Zero fd usage.
    Phase 2: write each flow to disk sequentially in 4MB chunks.
             Peak fds = 1 (one file open at a time).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Phase 1 accumulation
    flow_data  = {}        # key -> bytearray  (holds pcap header + all records)
    flow_order = []        # preserves insertion order for sequential naming
    written    = 0
    sentinels  = 0
    batch      = []
    t0         = time.time()

    print(f"[Writer {worker_id}] phase 1 — accumulating in RAM", flush=True)

    while True:
        try:
            item = queue.get(timeout=5)
        except Exception:
            continue

        if item is None:
            sentinels += 1
            if sentinels >= total_readers: break
            continue

        batch.append(item)
        for _ in range(BATCH_SIZE - 1):
            try:
                nxt = queue.get_nowait()
                if nxt is None:
                    sentinels += 1
                    if sentinels >= total_readers: break
                else:
                    batch.append(nxt)
            except Exception:
                break

        for key, record in batch:
            if key not in flow_data:
                buf = bytearray(PCAP_GLOBAL_HEADER)   # prepend header on first sight
                flow_data[key]  = buf
                flow_order.append(key)
            flow_data[key] += record
            written += 1

        batch.clear()

        if sentinels >= total_readers: break

        if written % (PROGRESS_EVERY // 2) == 0:
            ram_mb = sum(len(v) for v in flow_data.values()) / 1e6
            print(f"[Writer {worker_id}] {written/1e6:.1f}M records | "
                  f"{len(flow_data):,} flows | {ram_mb:.0f} MB RAM", flush=True)

    p1_elapsed = time.time() - t0
    total_ram  = sum(len(v) for v in flow_data.values()) / 1e9
    print(f"[Writer {worker_id}] phase 1 done in {p1_elapsed:.0f}s — "
          f"{written:,} records | {len(flow_data):,} flows | {total_ram:.2f} GB RAM",
          flush=True)

    # Phase 2: write to disk, one file at a time (peak fds = 1)
    print(f"[Writer {worker_id}] phase 2 — writing {len(flow_order):,} files to disk", flush=True)
    t1 = time.time()

    for i, key in enumerate(flow_order):
        path = os.path.join(OUTPUT_DIR, f"flow_w{worker_id}_{i}.pcap")
        data = flow_data[key]

        with open(path, "wb") as fh:            # "wb" not "ab" — we have everything
            mv   = memoryview(data)             # zero-copy view
            pos  = 0
            size = len(data)
            while pos < size:
                end = min(pos + DISK_WRITE_CHUNK, size)
                fh.write(mv[pos:end])           # write in 4MB chunks
                pos  = end

        flow_data[key] = None                  # release RAM as we go

        if (i + 1) % 10_000 == 0:
            p2_elapsed = time.time() - t1
            rate_mb    = total_ram * 1e3 * ((i+1)/len(flow_order)) / max(p2_elapsed, 1)
            print(f"[Writer {worker_id}] wrote {i+1:,}/{len(flow_order):,} files | "
                  f"{rate_mb:.0f} MB/s", flush=True)

    print(f"[Writer {worker_id}] phase 2 done in {time.time()-t1:.0f}s", flush=True)

# ── sentinel watcher ──────────────────────────────────────────────────────────
def sentinel_watcher(done_counter, total_readers, queues):
    while done_counter.value < total_readers:
        time.sleep(0.5)
    for q in queues:
        q.put(None)
    print("[Sentinel] all readers done — writers draining", flush=True)

# ── main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(INPUT_PCAP):
        print(f"ERROR: not found: {INPUT_PCAP}"); sys.exit(1)

    file_size = os.path.getsize(INPUT_PCAP)
    n_readers = NUM_READERS if USE_DPKT else 1

    try:
        soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        soft = "?"

    print("=" * 62)
    print(f"Input        : {INPUT_PCAP} ({file_size/1e9:.1f} GB)")
    print(f"Output       : {OUTPUT_DIR}")
    print(f"Readers      : {n_readers}   Writers : {NUM_WRITERS}")
    print(f"ulimit -n    : {soft}   (fd usage: ~{NUM_WRITERS + n_readers + 20} total)")
    print(f"Strategy     : accumulate all in RAM → write each file once")
    print(f"Parser       : {'dpkt (fast)' if USE_DPKT else 'scapy — pip install dpkt!'}")
    print(f"Queue depth  : {QUEUE_MAXSIZE}/writer   Batch : {BATCH_SIZE}")
    print(f"Disk chunk   : {DISK_WRITE_CHUNK//1024//1024} MB/write")
    print("=" * 62)

    usable = file_size - 24
    chunk  = usable // n_readers
    ranges = [
        (0 if i == 0 else 24 + i * chunk,
         file_size if i == n_readers - 1 else 24 + (i + 1) * chunk)
        for i in range(n_readers)
    ]

    queues       = [mp.Queue(maxsize=QUEUE_MAXSIZE) for _ in range(NUM_WRITERS)]
    pause_ev     = mp.Event()
    stop_ev      = mp.Event()
    done_counter = mp.Value("i", 0)

    t0 = time.time()

    mp.Process(target=ram_watchdog, args=(pause_ev, stop_ev), daemon=True).start()
    mp.Process(target=sentinel_watcher, args=(done_counter, n_readers, queues), daemon=True).start()

    writers = [mp.Process(target=writer_worker, args=(i, queues[i], n_readers))
               for i in range(NUM_WRITERS)]
    for p in writers: p.start()

    readers = [mp.Process(target=reader,
                           args=(i, ranges[i][0], ranges[i][1], queues, pause_ev, done_counter))
               for i in range(n_readers)]
    for p in readers: p.start()

    for p in readers: p.join()
    for p in writers: p.join()

    elapsed = time.time() - t0
    total_flows = sum(
        len([f for f in os.listdir(OUTPUT_DIR) if f.startswith(f"flow_w{i}_")])
        for i in range(NUM_WRITERS)
    )
    print(f"\nDone in {elapsed/3600:.2f} h  |  "
          f"flows: {total_flows:,}  |  out: {OUTPUT_DIR}")