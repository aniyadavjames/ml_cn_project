#!/usr/bin/python3
#-*- coding:utf-8 -*-

import os
import sys
import shutil
import binascii
import random
import json
import tqdm
import scapy.all as scapy
from flowcontainer.extractor import extract
from multiprocessing import Pool, cpu_count
import resource

# 1. FIX: Increase system limits for 28-core parallel handles
soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (65535, hard))

random.seed(40)

# Configuration
word_dir = "corpora/"
word_name = "encrypted_burst.txt"
splitcap_exe = "tools/SplitCap.exe" 

def cut(obj, sec):
    """Slices hex strings into n-gram chunks"""
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
    """Converts raw hex to space-separated bigrams for BERT"""
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

def split_cap(pcap_path, pcap_file, pcap_name):
    """Splits 189GB PCAP using Mono + SplitCap"""
    output_path = os.path.join(pcap_path, "splitcap", pcap_name)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # Optimized: -p 1000 stays under kernel limits, -b 1000000 speeds up disk write
    cmd = f"mono {splitcap_exe} -p 1000 -b 1000000 -r {pcap_file} -s session -o {output_path}"
    print(f"🛠️  Splitting: {os.path.basename(pcap_file)}")
    os.system(cmd)
    return output_path

def get_burst_feature_worker(args):
    """Parallel Worker: Processes one file and returns bigram text"""
    label_pcap, payload_len = args
    burst_txt = ""
    
    # Skip ghost/empty files to prevent 'No data could be read' errors
    if not os.path.exists(label_pcap) or os.path.getsize(label_pcap) < 1000:
        return ""

    try:
        packets = scapy.rdpcap(label_pcap)
        if len(packets) == 0: return ""
        
        # Guard against tshark/IPv6 metadata errors
        try:
            feature_result = extract(label_pcap)
        except:
            return f"LOG_SKIP: Metadata error in {os.path.basename(label_pcap)}\n"

        if not feature_result: return ""

        for key in feature_result.keys():
            value = feature_result[key]
            if not hasattr(value, 'ip_lengths') or len(value.ip_lengths) != len(packets):
                continue
            
            # Extract direction (1 for outbound, -1 for inbound)
            packet_direction = [x // abs(x) if x != 0 else 1 for x in value.ip_lengths]
            burst_data_string = ''
            
            for i in range(len(packets)):
                raw_bytes = bytes(packets[i])
                data = binascii.hexlify(raw_bytes).decode()[:2*payload_len]
                
                if i == 0:
                    burst_data_string += data
                else:
                    if packet_direction[i] != packet_direction[i-1]:
                        # End of a burst, process and reset string
                        length = len(burst_data_string)
                        if length > 0:
                            for s in cut(burst_data_string, max(1, int(length / 2))):
                                burst_txt += bigram_generation(s) + '\n'
                            burst_txt += '\n'
                        burst_data_string = data
                    else:
                        burst_data_string += data
            
            # Final flush for the session
            if burst_data_string:
                for s in cut(burst_data_string, max(1, int(len(burst_data_string) / 2))):
                    burst_txt += bigram_generation(s) + '\n'
                burst_txt += '\n'
                
        return burst_txt
    except Exception as e:
        return f"LOG_ERROR: {os.path.basename(label_pcap)} | {str(e)[:40]}\n"

def pretrain_dataset_generation(pcap_path):
    """Main Orchestrator"""
    # Create necessary folders
    if not os.path.exists("dataset/splitcap"): os.makedirs("dataset/splitcap")
    if not os.path.exists(word_dir): os.makedirs(word_dir)

    # 1. Splitting Phase
    print(f"Scanning: {os.path.abspath(pcap_path)}")
    for _parent, _dirs, files in os.walk(pcap_path):
        for file in files:
            if file.endswith(('.pcap', '.pcapng')):
                full_path = os.path.join(_parent, file)
                split_cap("dataset", full_path, file.replace('.pcap','').replace('.pcapng',''))

    # 2. Collection Phase
    print(f" Starting Parallel Extraction on {cpu_count()} cores...")
    task_list = []
    for root, _, files in os.walk("dataset/splitcap"):
        for f in files:
            if f.endswith('.pcap'):
                task_list.append((os.path.join(root, f), 64))

    # 3. Multiprocessing Pool
    with Pool(processes=cpu_count()) as pool:
        # Results are handled as they arrive to save memory
        results = pool.imap_unordered(get_burst_feature_worker, task_list)
        
        with open(os.path.join(word_dir, word_name), 'a') as f:
            for res in results:
                if res.startswith("LOG_"):
                    print(res.strip())
                elif res:
                    f.write(res)

    print(f"Finished! Corpus saved to {word_dir}{word_name}")

if __name__ == '__main__':
    # Make sure your 182GB file is in a folder named 'pcap'
    source_pcap_folder = "pcap/" 
    pretrain_dataset_generation(source_pcap_folder)