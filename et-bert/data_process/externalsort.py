#!/usr/bin/python3
#-*- coding:utf-8 -*-

import os
import binascii
import hashlib
import scapy.all as scapy
from collections import defaultdict

# --- HYBRID CONFIG ---
IS_HPC = os.path.exists("/scratch/cse/phd/csz258233")
PROJECT_ROOT = "/scratch/cse/phd/csz258233/col7560/et-bert" if IS_HPC else os.getcwd() 
SOURCE_PCAP = os.path.join(PROJECT_ROOT, "pcap", "all_packets_first_100.pcap")
TMP_BUCKET_DIR = os.path.join(PROJECT_ROOT, "dataset/temp_buckets")
FINAL_TEXT = os.path.join(PROJECT_ROOT, "corpora", "encrypted_burst.txt")

# We use 128 buckets to ensure each one fits in RAM during Phase 2
NUM_BUCKETS = 128 if IS_HPC else 8 

def get_session_id(pkt):
    if not pkt.haslayer(scapy.IP): return None
    ip = pkt[scapy.IP]
    ips = sorted([ip.src, ip.dst])
    ports = sorted([pkt.sport if hasattr(pkt, 'sport') else 0, pkt.dport if hasattr(pkt, 'dport') else 0])
    return f"{ips[0]}_{ips[1]}_{ports[0]}_{ports[1]}_{ip.proto}"

def run_external_merge_sort():
    os.makedirs(TMP_BUCKET_DIR, exist_ok=True)
    
    # --- PHASE 1: THE SHARDING (DISTRIBUTION) ---
    print("📦 Phase 1: Distributing packets into buckets...")
    # Open all bucket handles
    handles = {i: open(os.path.join(TMP_BUCKET_DIR, f"bucket_{i}.tmp"), 'a') for i in range(NUM_BUCKETS)}
    
    with scapy.PcapReader(SOURCE_PCAP) as reader:
        for pkt in reader:
            sid = get_session_id(pkt)
            if not sid: continue
            
            # Deterministic bucket assignment
            bucket_id = int(hashlib.md5(sid.encode()).hexdigest(), 16) % NUM_BUCKETS
            
            # Data format: SessionID | Timestamp | Direction | Hex
            # (Direction is 1 if src matches first IP in sorted sid)
            direction = 1 if pkt[scapy.IP].src == sid.split('_')[0] else -1
            line = f"{sid}|{pkt.time}|{direction}|{binascii.hexlify(bytes(pkt)).decode()}\n"
            handles[bucket_id].write(line)
            
    for h in handles.values(): h.close()

    # --- PHASE 2: THE MERGE & GROUPING ---
    print("🔄 Phase 2: Merging buckets and grouping sessions...")
    with open(FINAL_TEXT, 'w') as f_out:
        for i in range(NUM_BUCKETS):
            bucket_path = os.path.join(TMP_BUCKET_DIR, f"bucket_{i}.tmp")
            if not os.path.exists(bucket_path): continue
            
            # Load entire bucket into RAM (Safe because it's only 1/128th of the data)
            current_bucket_data = defaultdict(list)
            with open(bucket_path, 'r') as f_in:
                for line in f_in:
                    sid, ts, dr, hex_val = line.strip().split('|')
                    current_bucket_data[sid].append((float(ts), int(dr), hex_val))
            
            # Process each session in this bucket
            for sid in current_bucket_data:
                # 1. Sort packets of THIS session by timestamp
                packets = sorted(current_bucket_data[sid], key=lambda x: x[0])
                
                # 2. Extract bursts for this specific session
                current_burst_hex = ""
                current_dir = packets[0][1]
                
                for ts, dr, hex_val in packets:
                    if dr != current_dir:
                        # Direction flip: Write the completed burst
                        f_out.write(bigram_generation(current_burst_hex) + "\n")
                        current_burst_hex = hex_val
                        current_dir = dr
                    else:
                        current_burst_hex += hex_val
                
                # Write final burst and a separator
                f_out.write(bigram_generation(current_burst_hex) + "\n\n")
            
            # Clear memory for next bucket
            current_bucket_data.clear()
            print(f"Finished Bucket {i}/{NUM_BUCKETS}")

def bigram_generation(hex_data, pkt_len=64):
    truncated = hex_data[:2*pkt_len]
    return " ".join([truncated[i:i+2] for i in range(0, len(truncated)-1, 2)])

if __name__ == '__main__':
    run_external_merge_sort()