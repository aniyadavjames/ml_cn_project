import os, time, struct, hashlib, heapq
import multiprocessing as mp

try:
    import dpkt
except ImportError:
    raise RuntimeError("dpkt required")

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

# ── CONFIG ─────────────────────────────────────────────
INPUT_PCAP     = "scratch/flow/all/all_packets_first_100.pcap"
OUTPUT_DIR     = "scratch/flows_split/allh"

NUM_WRITERS    = 7
NUM_READERS    = 1   # ✅ FIXED
QUEUE_MAXSIZE  = 500_000
RAM_LIMIT_PCT  = 88
BATCH_SIZE     = 512
FLUSH_PERCENT  = 0.05

_rec_struct = struct.Struct("<IIII")

PCAP_GLOBAL_HEADER = struct.pack("<IHHiIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)

# ── UTILS ──────────────────────────────────────────────
def make_record(ts, raw):
    s  = int(ts)
    us = int((ts - s) * 1e6)
    n  = len(raw)
    return _rec_struct.pack(s, us, n, n) + raw

def stable_hash(key):
    return hashlib.md5(str(key).encode()).hexdigest()

def stable_hash_int(key):
    return int(hashlib.md5(str(key).encode()).hexdigest(), 16)

# ── FLOW EXTRACTION ────────────────────────────────────
import socket
def extract_dpkt(ts, raw):
    try:
        eth = dpkt.ethernet.Ethernet(raw)
        ip  = eth.data
        if not isinstance(ip, dpkt.ip.IP):
            return None, None

        seg = ip.data
        if isinstance(seg, dpkt.tcp.TCP):
            sp, dp = seg.sport, seg.dport
        elif isinstance(seg, dpkt.udp.UDP):
            sp, dp = seg.sport, seg.dport
        else:
            sp, dp = 0, 0

        return (
            socket.inet_ntoa(ip.src),
            socket.inet_ntoa(ip.dst),
            sp, dp, ip.p
        ), make_record(ts, raw)

    except Exception:
        return None, None

# ── FLUSH ──────────────────────────────────────────────
def flush_top_flows(flow_data, flow_pkt_count, worker_id):
    if not flow_data:
        return

    n = max(1, int(len(flow_data) * FLUSH_PERCENT))
    top = heapq.nlargest(n, flow_pkt_count.items(), key=lambda x: x[1])

    print(f"[Writer {worker_id}] FLUSH {n} flows", flush=True)

    for key, _ in top:
        data = flow_data[key]
        fname = os.path.join(OUTPUT_DIR, f"{stable_hash(key)}.pcap")

        mode = "ab" if os.path.exists(fname) else "wb"

        with open(fname, mode) as f:
            if mode == "wb":
                f.write(data)
            else:
                f.write(data[24:])

        del flow_data[key]
        del flow_pkt_count[key]

# ── WRITER ─────────────────────────────────────────────
def writer_worker(worker_id, queue, total_readers):

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    flow_data = {}
    flow_pkt_count = {}

    written = 0
    sentinels = 0
    batch = []

    last_log = time.time()
    last_written = 0

    print(f"[Writer {worker_id}] started", flush=True)

    while True:
        try:
            item = queue.get(timeout=5)
        except:
            continue

        if item is None:
            sentinels += 1
            if sentinels >= total_readers:
                break
            continue

        batch.append(item)

        for _ in range(BATCH_SIZE - 1):
            try:
                nxt = queue.get_nowait()
                if nxt is None:
                    sentinels += 1
                else:
                    batch.append(nxt)
            except:
                break

        for key, record in batch:
            if key not in flow_data:
                flow_data[key] = bytearray(PCAP_GLOBAL_HEADER)
                flow_pkt_count[key] = 0

            flow_data[key] += record
            flow_pkt_count[key] += 1
            written += 1

        batch.clear()

        # Flush conditions
        if len(flow_data) > 500_000:
            flush_top_flows(flow_data, flow_pkt_count, worker_id)

        if HAS_PSUTIL and psutil.virtual_memory().percent > RAM_LIMIT_PCT:
            flush_top_flows(flow_data, flow_pkt_count, worker_id)

        # Logging
        now = time.time()
        if now - last_log > 2:
            dt = now - last_log
            rate = (written - last_written) / dt
            ram_mb = sum(len(v) for v in flow_data.values()) / 1e6

            print(f"[Writer {worker_id}] {written/1e6:.2f}M rec | "
                  f"{len(flow_data):,} flows | "
                  f"{rate/1000:.1f} Krec/s | {ram_mb:.0f} MB",
                  flush=True)

            last_log = now
            last_written = written

    # ── FINAL FLUSH WITH LOGGING ──
    print(f"[Writer {worker_id}] final flush", flush=True)

    total = len(flow_data)
    i = 0
    last_log = time.time()

    for key, data in flow_data.items():
        i += 1

        fname = os.path.join(OUTPUT_DIR, f"{stable_hash(key)}.pcap")
        mode = "ab" if os.path.exists(fname) else "wb"

        with open(fname, mode) as f:
            if mode == "wb":
                f.write(data)
            else:
                f.write(data[24:])

        if time.time() - last_log > 2:
            print(f"[Writer {worker_id}] final flush {i}/{total} "
                  f"({i/total*100:.1f}%)", flush=True)
            last_log = time.time()

# ── READER ─────────────────────────────────────────────
def reader(reader_id, queues, done_counter):

    count = 0
    last_log = time.time()
    last_count = 0

    print(f"[Reader {reader_id}] started", flush=True)

    with open(INPUT_PCAP, "rb") as f:
        f.seek(24)  # skip global header

        while True:
            hdr = f.read(16)
            if len(hdr) < 16:
                break

            ts_s, ts_us, cap_len, _ = _rec_struct.unpack(hdr)
            raw = f.read(cap_len)

            key, rec = extract_dpkt(ts_s + ts_us * 1e-6, raw)
            if key is None:
                continue

            queues[stable_hash_int(key) % len(queues)].put((key, rec))
            count += 1

            now = time.time()
            if now - last_log > 2:
                dt = now - last_log
                rate = (count - last_count) / dt

                print(f"[Reader {reader_id}] {count/1e6:.2f}M pkts | "
                      f"{rate/1000:.1f} Kpkt/s",
                      flush=True)

                last_log = now
                last_count = count

    print(f"[Reader {reader_id}] done | {count/1e6:.2f}M pkts", flush=True)

    with done_counter.get_lock():
        done_counter.value += 1

# ── SENTINEL ───────────────────────────────────────────
def sentinel_watcher(done_counter, total_readers, queues):
    while done_counter.value < total_readers:
        time.sleep(1)
    for q in queues:
        q.put(None)

# ── MAIN ───────────────────────────────────────────────
if __name__ == "__main__":

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    queues = [mp.Queue(maxsize=QUEUE_MAXSIZE) for _ in range(NUM_WRITERS)]
    done_counter = mp.Value("i", 0)

    mp.Process(target=sentinel_watcher,
               args=(done_counter, NUM_READERS, queues),
               daemon=True).start()

    writers = [mp.Process(target=writer_worker,
                          args=(i, queues[i], NUM_READERS))
               for i in range(NUM_WRITERS)]

    for p in writers:
        p.start()

    reader_p = mp.Process(target=reader,
                          args=(0, queues, done_counter))

    reader_p.start()
    reader_p.join()

    print("\n[MAIN] Reading complete. Writers finishing...\n", flush=True)

    for p in writers:
        p.join()

    print("\nDone.")