#!/usr/bin/env python
# coding: utf-8

# ## Manifold ID calculation

# In[1]:


import os
import numpy as np
import pickle
from dadapy.data import Data
import torch

def calculate_id(filepath: str):
    with open(filepath, "rb") as f:
        embeddings, filenames = pickle.load(f)
    assert isinstance(embeddings, torch.Tensor)
    nantonum = lambda x: torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    embeddings = nantonum(embeddings).numpy()
    embeddings = np.nan_to_num(embeddings, nan=0, posinf=0, neginf=0)
    dataset = Data(embeddings)
    dataset.remove_identical_points()
    dataset.compute_distances(maxk=100)
    ids = dataset.compute_id_2NN()
    print(f"File: {filepath}, result: {ids}")


# ### YaTC

# In[ ]:


directory = '../data/yatc/'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    calculate_id(filepath)


# ### ET-BERT

# In[ ]:


directory = '../data/etbert/'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    calculate_id(filepath)


# ### netFound

# In[ ]:


directory = '../data/netfound/'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    calculate_id(filepath)


# ### NetMamba

# In[ ]:


directory = '../data/netmamba/'
for filename in os.listdir(directory):
    filepath = os.path.join(directory, filename)
    calculate_id(filepath)

