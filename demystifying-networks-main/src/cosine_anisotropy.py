#!/usr/bin/env python
# coding: utf-8

# ## Cosine similarity anisotropy

# In[1]:


import os
import torch
import torch.nn.functional as F
import pickle
import numpy as np
import random
import argparse
from tqdm import tqdm


# In[2]:


def cos_contrib(emb1, emb2):
    numerator_terms = emb1 * emb2
    denom = np.linalg.norm(emb1) * np.linalg.norm(emb2)
    return np.array(numerator_terms / denom)


def measure_anisotropy(filepath):
    with open(filepath, "rb") as f:
        embeddings, _ = pickle.load(f)
    
    indices = torch.randperm(embeddings.size(0))
    embeddings = embeddings[indices]
    
    layer_cosine_contribs = []

    for i in tqdm(range(embeddings.shape[0] - 1)):
        emb1, emb2 = embeddings[i, :], embeddings[i+1, :]
        layer_cosine_contribs.append(cos_contrib(emb1, emb2))
    
    layer_cosine_contribs = np.stack(layer_cosine_contribs)
    layer_cosine_contribs_mean = layer_cosine_contribs.mean(axis=0)
    
    aniso = layer_cosine_contribs_mean.sum()    
    top_dims = np.argsort(layer_cosine_contribs_mean)[-10:]
    top_dims = np.flip(top_dims)
    
    print(f"### {filepath} ###")
    print(f"Top 10 dims: {top_dims}")
    print(f"Estimated anisotropy: {aniso}")
    for i in range(10):
        d = top_dims[i]
        print(d, layer_cosine_contribs_mean[d])


# ### YaTC

# In[3]:


directory = '../data/yatc/'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    result = measure_anisotropy(filepath)
    print(f"{filename} - {result}")


# ### ET-BERT

# In[4]:


directory = '../data/etbert/'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    result = measure_anisotropy(filepath)
    print(f"{filename} - {result}")


# ### netFound

# In[5]:


directory = '../data/netfound/'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    result = measure_anisotropy(filepath)
    print(f"{filename} - {result}")


# ### netMamba

# In[6]:


directory = '../data/netmamba/'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    result = measure_anisotropy(filepath)
    print(f"{filename} - {result}")

