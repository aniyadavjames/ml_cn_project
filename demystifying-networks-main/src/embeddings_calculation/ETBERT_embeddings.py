import os
import sys
import torch
import argparse
import threading
import pickle
import copy
from tqdm import tqdm
from collections import defaultdict

# --- 1. DYNAMIC PATH SETUP ---
# Get the absolute path of the directory where this script resides
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Based on your path: ~/Desktop/project/ml_cn_project/demystifying-networks-main/models/ET-BERT/src
# We go up two levels from 'embeddings_calculation' to 'src', 
# then into 'models/ET-BERT/src'
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../models/ET-BERT/"))
# Add it to sys.path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
print(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
print(PROJECT_ROOT)
# Verify the path and try imports
try:
    from finetuning.run_classifier import Classifier, read_dataset, load_or_initialize_parameters, count_labels_num, batch_loader
    from uer.opts import finetune_opts
    from uer.utils.config import load_hyperparam
    from uer.utils.constants import *
    from uer.utils import *
    from uer.utils.optimizers import *
    print(f"✅ Success: ET-BERT modules loaded from {PROJECT_ROOT}")
except ImportError as e:
    print(f"Critical Error: Could not find ET-BERT modules at {PROJECT_ROOT}")
    print("Check if 'finetuning' and 'uer' folders exist in that directory.")
    sys.exit(1)

# --- 2. THE PIPELINE ENGINE ---
def get_embeddings(datafolder, batch_size=64, 
                   pretrained_model="/dev/shm/pretrained_model_etbert.bin", 
                   limit=10**30):
    
    # Environment Check
    gpus = torch.cuda.device_count()
    if gpus > 0:
        print(f"🚀 HPC Mode: Using {gpus} GPUs.")
        device = torch.device("cuda:0")
    else:
        print("💻 Local Mode: No GPU found. Using CPU (will be slow).")
        gpus = 1
        device = torch.device("cpu")

    # Argument Parsing for ET-BERT
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)
    
    # Map paths correctly
    vocab_path = os.path.join(PROJECT_ROOT, "models/encryptd_vocab.txt")
    train_path = os.path.join(datafolder, "train_dataset.tsv")
    
    args = parser.parse_args([
        "--train_path", train_path,
        "--vocab_path", vocab_path,
        "--dev_path", train_path, # Dummy for initialization
        "--pretrained_model_path", pretrained_model
    ])
# --- 3. EXECUTION BLOCK ---
if __name__ == "__main__":
    # 1. Define where your data folders are
    # Based on your structure, they are likely in ../../data/ relative to the script
    DATA_BASE_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/"))
    
    # 2. List the datasets you want to process
    target_datasets = ["cross", "caida", "cicapt", "cicids", "mawi", "etbert"]
    
    print(f"📂 Starting pipeline. Looking for data in: {DATA_BASE_PATH}")

    for dataset_name in target_datasets:
        folder_path = os.path.join(DATA_BASE_PATH, dataset_name)
        
        if os.path.exists(folder_path):
            print(f"\n🔄 Processing Dataset: {dataset_name}")
            try:
                # 3. CALL the function you defined earlier
                embeddings = get_embeddings(
                    datafolder=folder_path,
                    batch_size=512 if torch.cuda.is_available() else 16, # Use 512 for HPC, 16 for Local
                    pretrained_model="/dev/shm/pretrained_model_etbert.bin" # Ensure this path is correct on HPC
                )
                
                # 4. Save the results
                save_file = os.path.join(DATA_BASE_PATH, f"{dataset_name}_emb.pkl")
                with open(save_file, "wb") as f:
                    pickle.dump(embeddings, f)
                print(f"✅ Saved embeddings to {save_file}")
                
            except Exception as e:
                print(f"❌ Failed to process {dataset_name}: {e}")
        else:
            print(f"⚠️ Skipping {dataset_name}: Folder not found at {folder_path}")