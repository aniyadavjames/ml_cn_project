#!/usr/bin/python3
#-*- coding:utf-8 -*-

import os
import sys
import shutil
import binascii
import random
import json
import tqdm
import resource
from multiprocessing import Pool, cpu_count

# --- 1. HYBRID CONFIGURATION SWITCH ---
# This detects if we are on the IITD cluster or your laptop/PC
IS_HPC = os.path.exists("/scratch/cse/phd/csz258233")

if IS_HPC:
    # HPC PATHS
    env_site_packages = "/scratch/cse/phd/csz258233/py39_env/lib/python3.9/site-packages"
    PROJECT_ROOT = "/scratch/cse/phd/csz258233/col7560/et-bert"
    SOURCE_PCAP_DIR = "/scratch/cse/phd/csz258233/col7560/et-bert/pcaps"
    # For HPC, use tcpdump because mono is usually missing
    SPLIT_COMMAND_TEMPLATE = "tcpdump -r {input} -w {output}/session -C 100"
else:
    # LOCAL PATHS (Change these to match your personal computer)
    env_site_packages = None # Local env is usually handled by conda/venv automatically
    PROJECT_ROOT = os.getcwd() # Uses the folder where the script is
    SOURCE_PCAP_DIR = os.path.join(PROJECT_ROOT, "pcap")
    # For Local, use SplitCap.exe if you are on Windows, or tcpdump if on Mac/Linux
    SPLIT_COMMAND_TEMPLATE = "tcpdump -r {input} -w {output}/session -C 10" 

# Add custom environment to path if on HPC
if IS_HPC and env_site_packages and env_site_packages not in sys.path:
    sys.path.insert(0, env_site_packages)

# Now safe to import network libraries
import scapy.all as scapy
from flowcontainer.extractor import extract

# Global Constants
OUTPUT_BASE = os.path.join(PROJECT_ROOT, "dataset/splitcap")
CORPUS_DIR = os.path.join(PROJECT_ROOT, "corpora")
WORD_NAME = "encrypted_burst.txt"

random.seed(40)

# --- 2. SYSTEM UTILITIES ---

def maximize_file_limits():
    """Optimizes system for high-concurrency file reading"""
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        # On local machines, 'hard' might be small; on HPC it's large.
        # We set soft to hard to avoid ValueError.
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        new_soft, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if IS_HPC: print(f"DEBUG: File limits maximized to {new_soft}")
    except Exception as e:
        print(f"Note: Limit maximization skipped ({e})")

def cut(obj, sec):
    if not obj: return []
    result = [obj[i:i+sec] for i in range(0, len(obj), sec)]
    try:
        remanent_count = len(result[0]) % 4
    except:
        remanent_count = 0
    if remanent_count != 0:
        result = [obj[i:i+sec+remanent_count] for i in range(0, len(obj), sec+remanent_count)]
    return result

def bigram_generation(packet_datagram, packet_len=64):
    result = ''
    generated_datagram = cut(packet_datagram, 1)
    token_count = 0
    for i in range(len(generated_datagram)):
        if i != (len(generated_datagram) - 1):
            token_count += 1
            if token_count > packet_len: break
            merge_word = generated_datagram[i] + generated_datagram[i + 1]
            result += merge_word + ' '
        else: break
    return result

# --- 3. PROCESSING FUNCTIONS ---

def split_cap_logic(full_path, pcap_name):
    """Handles the splitting phase based on the template in config"""
    output_path = os.path.join(OUTPUT_BASE, pcap_name)
    os.makedirs(output_path, exist_ok=True)
    
    # Fill the template with actual paths
    cmd = SPLIT_COMMAND_TEMPLATE.format(input=full_path, output=output_path)
    
    print(f"🛠️  Splitting: {os.path.basename(full_path)}")
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"ERROR: Split command failed for {pcap_name}")
    return output_path

def get_burst_feature_worker(args):
    """Worker function for parallel core usage"""
    label_pcap, payload_len = args
    burst_txt = ""
    
    if not os.path.exists(label_pcap) or os.path.getsize(label_pcap) < 100:
        return ""

    try:
        packets = scapy.rdpcap(label_pcap)
        if len(packets) == 0: return ""
        
        try:
            feature_result = extract(label_pcap)
        except:
            return ""

        if not feature_result: return ""

        for key in feature_result.keys():
            value = feature_result[key]
            if not hasattr(value, 'ip_lengths') or len(value.ip_lengths) != len(packets):
                continue
            
            packet_direction = [x // abs(x) if x != 0 else 1 for x in value.ip_lengths]
            burst_data_string = ''
            
            for i in range(len(packets)):
                raw_bytes = bytes(packets[i])
                data = binascii.hexlify(raw_bytes).decode()[:2*payload_len]
                
                if i == 0:
                    burst_data_string += data
                else:
                    if packet_direction[i] != packet_direction[i-1]:
                        length = len(burst_data_string)
                        if length > 0:
                            for s in cut(burst_data_string, max(1, int(length / 2))):
                                burst_txt += bigram_generation(s) + '\n'
                            burst_txt += '\n'
                        burst_data_string = data
                    else:
                        burst_data_string += data
            
            if burst_data_string:
                for s in cut(burst_data_string, max(1, int(len(burst_data_string) / 2))):
                    burst_txt += bigram_generation(s) + '\n'
                burst_txt += '\n'
                
        return burst_txt
    except Exception:
        return ""

# --- 4. MAIN ORCHESTRATOR ---

def pretrain_dataset_generation(pcap_path):
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(CORPUS_DIR, exist_ok=True)

    print(f"--- Mode: {'HPC' if IS_HPC else 'LOCAL'} ---")
    print(f"DEBUG: Scanning Source: {pcap_path}")
    
    pcap_files = []
    for _parent, _dirs, files in os.walk(pcap_path):
        for file in files:
            if file.lower().endswith(('.pcap', '.pcapng')):
                pcap_files.append(os.path.join(_parent, file))

    if not pcap_files:
        print("CRITICAL: No PCAP files found. Check your SOURCE_PCAP_DIR.")
        return

    # Phase 1: Split
    for full_path in pcap_files:
        name = os.path.basename(full_path).replace('.pcap','').replace('.pcapng','')
        split_cap_logic(full_path, name)

    # Phase 2: Collect session chunks
    print(f"Starting Extraction on {cpu_count()} cores...")
    task_list = []
    for root, _, files in os.walk(OUTPUT_BASE):
        for f in files:
            # Matches 'session', 'session0', etc. from tcpdump
            if 'session' in f:
                task_list.append((os.path.join(root, f), 64))

    # Phase 3: Parallel Process
    if task_list:
        with Pool(processes=cpu_count()) as pool:
            results = pool.imap_unordered(get_burst_feature_worker, task_list)
            
            output_file = os.path.join(CORPUS_DIR, WORD_NAME)
            with open(output_file, 'a') as f:
                for res in results:
                    if res: f.write(res)
    else:
        print("DEBUG: Task list empty. Nothing to extract.")

    print(f"Finished! Output: {os.path.join(CORPUS_DIR, WORD_NAME)}")

if __name__ == '__main__':
    maximize_file_limits()
    pretrain_dataset_generation(SOURCE_PCAP_DIR)