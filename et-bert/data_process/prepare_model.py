# Save as prepare_model.py
from transformers import BertTokenizer, BertForMaskedLM
import os

# Confirming the script sees your exported proxy
print(f"Current HTTP Proxy: {os.environ.get('http_proxy')}")

try:
    # 1. Download Base BERT (uses the exported proxy)
    model_name = "bert-base-uncased"
    print(f"Downloading {model_name}...")
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)

    # 2. Add Hex Bigram Tokens (00 to ff)
    # This is critical for ET-BERT to understand your PCAP bigrams
    hex_tokens = [f"{i:02x}" for i in range(256)]
    tokenizer.add_tokens(hex_tokens)
    
    # 3. Resize the model embeddings to accommodate the 256 new tokens
    model.resize_token_embeddings(len(tokenizer))

    # 4. Save to Scratch (to use later on the offline GPU nodes)
    save_path = "/scratch/cse/phd/csz258233/col7560/et-bert/base_model"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)
    
    print(f"\n--- SUCCESS ---")
    print(f"Model and Tokenizer with 256 hex tokens saved to: {save_path}")
    print(f"Vocab size is now: {len(tokenizer)}")

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"If this fails with 'Name or service not known', your bash exports")
    print(f"didn't pass to Python. Detailed error: {e}")
