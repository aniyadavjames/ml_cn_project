#!/usr/bin/env python
# coding: utf-8

# ## netFound embeddings

# In[ ]:


get_ipython().run_line_magic('cd', '../models/netFound/src/train')


# In[5]:


import torch
import torch.nn
import threading
from collections import defaultdict
from NetFoundModels import NetFoundLanguageModelling, NetfoundFinetuningModel, NetFoundBase
from NetfoundConfig import NetFoundLarge
from NetfoundTokenizer import NetFoundTokenizer
from NetFoundDataCollator import SimpleDataCollator
from NetFoundTrainer import NetfoundTrainer
from torch.utils.data import DataLoader
from transformers import Trainer, TrainingArguments

from datasets import load_dataset
import pickle
from tqdm import tqdm
import copy
import threading
from collections import defaultdict


# In[ ]:


def load_model(path: str):
    config = NetFoundLarge()
    return NetfoundFinetuningModel.from_pretrained(path, config=config, ignore_mismatched_sizes=True).to("cpu")

model = load_model("../models/netFound/netfound_checkpoint")
models = {}
for i in range(4):
    models[i] = copy.deepcopy(model)
    models[i].to(f"cuda:{i}")
del model


# In[7]:


def load_data(path, batch_size = 32):
    tdataset = load_dataset("arrow", data_dir=path, split="train", cache_dir="/tmp/tmp", streaming=False)
    total_bursts_train = [0] * len(tdataset)
    tdataset = tdataset.add_column("total_bursts", total_bursts_train)
    
    config = NetFoundLarge()
    config.pretraining = True
    tokenizer = NetFoundTokenizer(config=config)
    tokenizer.raw_labels = True
    
    def preprocess_function(examples):
        return tokenizer(examples)

    tdataset = tdataset.map(preprocess_function, batched=True, num_proc=110, load_from_cache_file=True)

    data_loader = torch.utils.data.DataLoader(
        tdataset.remove_columns(['burst_tokens', 'directions', 'counts']),
        batch_size=batch_size,
        num_workers=8,
        prefetch_factor=2,
        collate_fn=SimpleDataCollator(config.max_burst_length),
        drop_last=True,
    )
    return tdataset, data_loader

def encode(batch, model):
    with torch.no_grad():
        batch['position_ids'] = torch.arange(
                batch['input_ids'].size(1),
                device=batch['input_ids'].device
            ).unsqueeze(0).expand(batch['input_ids'].size(0), -1)

        output = model.base_transformer(
            input_ids=batch['input_ids'].to(model.device),
            attention_mask=batch['attention_mask'].to(model.device),
            position_ids=batch['position_ids'].to(model.device),
            direction=batch['direction'].to(model.device),
            iats=batch['iats'].to(model.device),
            bytes=batch['bytes'].to(model.device),
            return_dict=True,
            pkt_count=batch['pkt_count'].to(model.device),
            protocol=batch['protocol'].to(model.device),
        ).last_hidden_state
        return torch.mean(output, 1).cpu(), batch["labels"]

def encode_and_append(batch, model, result_list, i):
    with torch.no_grad():
        result_list.append(encode(batch, model))

def get_embeddings(datafolder, models, batch_size=64, limit = 10**30, gpus=4):
    _, dataloader = load_data(datafolder, batch_size=batch_size)
    print(f"Total: {len(dataloader)}")
    
    counter = 0
    result_embeddings = []
    result_filenames = []

    with torch.no_grad():
        iterator = iter(dataloader)
        try:
            for y in tqdm(range(0, min(len(dataloader) // gpus, limit))):
                emb = defaultdict(list)
                batches = [next(iterator) for i in range(gpus)]
                threads = []
                for i in range(gpus):
                    t = threading.Thread(target=encode_and_append, args=(batches[i], models[i], emb[i], i))
                    t.start()
                    threads.append(t)
                for t in threads:
                    t.join()
                del batches
                result_embeddings.extend([emb[i][0][0] for i in emb])
                for i in emb:
                    result_filenames.extend(emb[i][0][1])
        except StopIteration:
            print("finished")
        except Exception as e:
            print(e)
    return torch.cat(result_embeddings), result_filenames


# In[8]:


labels = ["synth"]


# In[ ]:


for label in labels:
    embeddings = get_embeddings(f"../data/{label}", models, batch_size=1, gpus=1)
    with open(f"../data/{label}_emb.pkl", "bw") as f:
        pickle.dump(embeddings, f)


# In[ ]:




