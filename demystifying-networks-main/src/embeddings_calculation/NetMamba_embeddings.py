#!/usr/bin/env python
# coding: utf-8

# ## netmamba embeddings

# In[ ]:


get_ipython().run_line_magic('cd', '../models/NetMamba/src')


# In[3]:


from torchvision import datasets, transforms
import torch
import models_net_mamba
from util.pos_embed import interpolate_pos_embed
from timm.models.layers import trunc_normal_
import os
from tqdm import tqdm
import copy
from collections import defaultdict
import threading
import pickle


# In[4]:


os.environ['PATH'] = '/sbin:' + os.environ.get('PATH', '')


# In[ ]:


def build_dataset(data_path):
    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return dataset

def get_embeddings(datafolder, batch_size=64, limit = 10**30, gpus=4):
    dataset = build_dataset(datafolder)
    sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=8
    )
    model = models_net_mamba.__dict__['net_mamba_classifier'](
        num_classes=2,
        drop_path_rate=0,
    )

    checkpoint = torch.load('../models/NetMamba/pre-train.pth', map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]
    
    # interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)
    
    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    
    # manually initialize fc layer
    trunc_normal_(model.head.weight, std=2e-5)

    models = {}
    for i in range(gpus):
        models[i] = copy.deepcopy(model)
        models[i].to(f"cuda:{i}")
    
    def encode_and_append(batch, model, result_list, i):
        imgs, _ = batch
        with torch.no_grad():
            result_list.append(model.forward_encoder(imgs.to(f"cuda:{i}"), mask_ratio=0.0, if_mask=False)[:, -1, :].cpu())
        del imgs
        del batch

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
                result_embeddings.append(torch.cat([torch.cat(emb[i]) for i in emb]))
                result_filenames.extend([x[0] for x in dataloader.dataset.samples[y*gpus*batch_size:(y+1)*gpus*batch_size]])
                
        except StopIteration:
            print("finished")
        except Exception as e:
            print(e)

    return torch.cat(result_embeddings), result_filenames
    


# In[7]:


labels = ['cross', 'cicids', 'cicapt', 'caida', 'mawi']


# In[8]:


labels = ['synth']


# In[ ]:


for label in labels:
    print(label)
    emb = get_embeddings(f"../data/{label}/array_sampled", batch_size=512)
    with open(f"../data/netmamba_{label}_emb.pkl", "bw") as f:
        pickle.dump(emb, f)


# In[ ]:


label = "synth"
emb = get_embeddings(f"../data/{label}/array_sampled", batch_size=1, gpus=1)
with open(f"../data/netmamba_{label}_emb.pkl", "bw") as f:
    pickle.dump(emb, f)


# In[ ]:




