import os
import binascii
from scapy.all import rdpcap, IP
import multiprocessing as mp

# --- CONFIG ---
INPUT_DIR = "/scratch/cse/phd/csz258231/flows_split/final_flows"
TMP_DIR = "/scratch/cse/phd/csz258233/col7560/et-bert/tmp_output"
PAYLOAD_LEN = 64  # ET-BERT standard: first 64 bytes of packet

os.makedirs(TMP_DIR, exist_ok=True)

def process_hex_to_bigrams(hex_str):
    """The overlapping bigram logic: AB BC CD"""
    # hex_str is a string like '450000...' -> split into bytes '45', '00', '00'
    res = [hex_str[i:i+2] for i in range(0, len(hex_str), 2)]
    bigrams = [res[i] + res[i+1] for i in range(len(res) - 1)]
    return " ".join(bigrams)

def process_pcap_batch(file_list, worker_id):
    """Worker function to process a list of session PCAPs"""
    output_path = os.path.join(TMP_DIR, f"worker_{worker_id}.txt")
    
    with open(output_path, "w") as f_out:
        for pcap_file in file_list:
            try:
                pcap_path = os.path.join(INPUT_DIR, pcap_file)
                packets = rdpcap(pcap_path)
                if not packets: continue
                
                current_burst_hex = ""
                last_dir = None
                # Assume first packet's source is the "Client" for directionality
                client_ip = packets[0][IP].src if packets[0].haslayer(IP) else None

                for pkt in packets:
                    if not pkt.haslayer(IP): continue
                    
                    # Determine direction relative to the first packet
                    current_dir = 1 if pkt[IP].src == client_ip else -1
                    
                    # Extract Packet Data (Strip Ethernet 14 bytes)
                    # We take 14:14+PAYLOAD_LEN to get the IP header + start of payload
                    raw_hex = binascii.hexlify(bytes(pkt)[14:14+PAYLOAD_LEN]).decode()
                    
                    if last_dir is not None and current_dir != last_dir:
                        # Direction changed: Process the completed burst
                        if current_burst_hex:
                            f_out.write(process_hex_to_bigrams(current_burst_hex) + "\n")
                        current_burst_hex = raw_hex
                    else:
                        # Same direction: Append to current burst
                        current_burst_hex += raw_hex
                    
                    last_dir = current_dir
                
                # Write the final burst of the session
                if current_burst_hex:
                    f_out.write(process_hex_to_bigrams(current_burst_hex) + "\n")
                
                # Add a blank line to separate this session from the next in the corpus
                f_out.write("\n")
                
            except Exception:
                # Skip corrupted PCAPs or non-IP traffic
                continue

if __name__ == "__main__":
    # Get all PCAP files from the directory
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.pcap')]
    num_workers = 28  # IIT Delhi HPC standard core count
    
    # Chunking the file list
    avg = len(all_files) // num_workers
    chunks = [all_files[i:i + avg] for i in range(0, len(all_files), avg)]

    print(f"Distributing {len(all_files)} sessions across {num_workers} processors...")
    
    # Launching processes
    processes = []
    for i in range(num_workers):
        p = mp.Process(target=process_pcap_batch, args=(chunks[i], i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

<<<<<<< HEAD
    print("Processing complete. Ready to merge.")
=======
    print("Processing complete. Ready to merge.")
>>>>>>> 1ae286efdce9d7b138f185dfdc6f84bde9d216d3
