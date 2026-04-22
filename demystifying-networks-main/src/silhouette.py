#!/usr/bin/env python
# coding: utf-8

# ## Silhouette calculation

# In[4]:


from sklearn.metrics import silhouette_score
import pickle
import torch
import os


# In[5]:


def calculate_silhouette(filepath, filename_mapping):
    with open(filepath, "rb") as f:
        embeddings, filenames = pickle.load(f)

    filenames = [filename_mapping(x) for x in filenames]  # filepaths to labels
    score = silhouette_score(embeddings, filenames)
    print(f"{filepath} - silhouette score: {score}")


# ### YaTC

# In[6]:


def mapping(x):
    return x.removeprefix("/dev/shm/data/").split('/')[2]

directory = '../data/yatc/'
for filename in os.listdir(directory):
    if (
        "mawi" in filename or
        "cicapt" in filename or
        "caida" in filename
    ):
        # no labels for these datasets
        continue
    filepath = os.path.join(directory, filename)
    calculate_silhouette(filepath, mapping)


# ### ET-BERT

# In[ ]:


def mapping(x):
    return [i for i in x.removeprefix("/dev/shm/data/").removeprefix("/dev/shm/data2/").split('/') if i != ''][1]

directory = '../data/etbert/'
for filename in os.listdir(directory):
    if (
        "mawi" in filename or
        "cicapt" in filename or
        "caida" in filename
    ):
        # no labels for these datasets
        continue
    filepath = os.path.join(directory, filename)
    calculate_silhouette(filepath, mapping)


# ### netFound

# In[ ]:


def calculate_silhouette_nf(filepath, filename_mapping):
    with open(filepath, "rb") as f:
        embeddings, filenames = pickle.load(f)

    # filename mapping is different for different datasets in netfound
    filenames = [filename_mapping(filepath, x) for x in filenames]  # filepaths to labels
    score = silhouette_score(embeddings, filenames)
    print(f"{filepath} - silhouette score: {score}")

def mapping_nf(filepath, x):
    if "cross_emb.pkl" in filepath:
        return x.removeprefix("/data/").split('-')[0]
    elif "cicapt_emb.pkl" in filepath:
        return '-'.join(x.removeprefix("/data/").split('-')[:3])
    elif "cicids_emb.pkl" in filepath:
        return '-'.join(x.removeprefix('/data/').split('_')[1:3]).split('.')[0]
    return ""

directory = '/dev/shm/data/netfound/'
for filename in os.listdir(directory):
    if (
        "mawi" in filename or
        "cicapt" in filename or
        "caida" in filename
    ):
        # no labels for these datasets
        continue
    filepath = os.path.join(directory, filename)
    calculate_silhouette_nf(filepath, mapping_nf)


# ### netMamba

# In[ ]:


def mapping(x):
    return x.removeprefix("/dev/shm/data/").split('/')[2]

directory = '../data/netmamba/'
for filename in os.listdir(directory):
    if (
        "mawi" in filename or
        "cicapt" in filename or
        "caida" in filename
    ):
        # no labels for these datasets
        continue
    filepath = os.path.join(directory, filename)
    calculate_silhouette(filepath, mapping)


# In[ ]:




