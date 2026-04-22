#!/usr/bin/env python
# coding: utf-8

# # netFound feature perturbation

# In[1]:


import numpy as np

# define CKA

def gram_linear(x):
    """Compute Gram (kernel) matrix for a linear kernel.

    Args:
        x: A num_examples x num_features matrix of features.

    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    return x.dot(x.T)


def gram_rbf(x, threshold=1.0):
    """Compute Gram (kernel) matrix for an RBF kernel.

    Args:
        x: A num_examples x num_features matrix of features.
        threshold: Fraction of median Euclidean distance to use as RBF kernel
        bandwidth. (This is the heuristic we use in the paper. There are other
        possible ways to set the bandwidth; we didn't try them.)

    Returns:
        A num_examples x num_examples Gram matrix of examples.
    """
    dot_products = x.dot(x.T)
    sq_norms = np.diag(dot_products)
    sq_distances = -2 * dot_products + sq_norms[:, None] + sq_norms[None, :]
    sq_median_distance = np.median(sq_distances)
    return np.exp(-sq_distances / (2 * threshold ** 2 * sq_median_distance))


def center_gram(gram, unbiased=False):
    """Center a symmetric Gram matrix.

    This is equvialent to centering the (possibly infinite-dimensional) features
    induced by the kernel before computing the Gram matrix.

    Args:
        gram: A num_examples x num_examples symmetric matrix.
        unbiased: Whether to adjust the Gram matrix in order to compute an unbiased
        estimate of HSIC. Note that this estimator may be negative.

    Returns:
        A symmetric matrix with centered columns and rows.
    """
    if not np.allclose(gram, gram.T):
        raise ValueError('Input must be a symmetric matrix.')
    gram = gram.copy()

    if unbiased:
        # This formulation of the U-statistic, from Szekely, G. J., & Rizzo, M.
        # L. (2014). Partial distance correlation with methods for dissimilarities.
        # The Annals of Statistics, 42(6), 2382-2412, seems to be more numerically
        # stable than the alternative from Song et al. (2007).
        n = gram.shape[0]
        np.fill_diagonal(gram, 0)
        means = np.sum(gram, 0, dtype=np.float64) / (n - 2)
        means -= np.sum(means) / (2 * (n - 1))
        gram -= means[:, None]
        gram -= means[None, :]
        np.fill_diagonal(gram, 0)
    else:
        means = np.mean(gram, 0, dtype=np.float64)
        means -= np.mean(means) / 2
        gram -= means[:, None]
        gram -= means[None, :]

    return gram


def cka(gram_x, gram_y, debiased=False):
    """Compute CKA.

    Args:
        gram_x: A num_examples x num_examples Gram matrix.
        gram_y: A num_examples x num_examples Gram matrix.
        debiased: Use unbiased estimator of HSIC. CKA may still be biased.

    Returns:
        The value of CKA between X and Y.
    """
    gram_x = center_gram(gram_x, unbiased=debiased)
    gram_y = center_gram(gram_y, unbiased=debiased)

    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)


def _debiased_dot_product_similarity_helper(
    xty, sum_squared_rows_x, sum_squared_rows_y, squared_norm_x, squared_norm_y,
    n):
  """Helper for computing debiased dot product similarity (i.e. linear HSIC)."""
  # This formula can be derived by manipulating the unbiased estimator from
  # Song et al. (2007).
  return (
      xty - n / (n - 2.) * sum_squared_rows_x.dot(sum_squared_rows_y)
      + squared_norm_x * squared_norm_y / ((n - 1) * (n - 2)))


def feature_space_linear_cka(features_x, features_y, debiased=False):
    """Compute CKA with a linear kernel, in feature space.

    This is typically faster than computing the Gram matrix when there are fewer
    features than examples.

    Args:
        features_x: A num_examples x num_features matrix of features.
        features_y: A num_examples x num_features matrix of features.
        debiased: Use unbiased estimator of dot product similarity. CKA may still be
        biased. Note that this estimator may be negative.

    Returns:
        The value of CKA between X and Y.
    """
    features_x = features_x - np.mean(features_x, 0, keepdims=True)
    features_y = features_y - np.mean(features_y, 0, keepdims=True)

    dot_product_similarity = np.linalg.norm(features_x.T.dot(features_y)) ** 2
    normalization_x = np.linalg.norm(features_x.T.dot(features_x))
    normalization_y = np.linalg.norm(features_y.T.dot(features_y))

    if debiased:
        n = features_x.shape[0]
        # Equivalent to np.sum(features_x ** 2, 1) but avoids an intermediate array.
        sum_squared_rows_x = np.einsum('ij,ij->i', features_x, features_x)
        sum_squared_rows_y = np.einsum('ij,ij->i', features_y, features_y)
        squared_norm_x = np.sum(sum_squared_rows_x)
        squared_norm_y = np.sum(sum_squared_rows_y)

        dot_product_similarity = _debiased_dot_product_similarity_helper(
            dot_product_similarity, sum_squared_rows_x, sum_squared_rows_y,
            squared_norm_x, squared_norm_y, n)
        normalization_x = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_x ** 2, sum_squared_rows_x, sum_squared_rows_x,
            squared_norm_x, squared_norm_x, n))
        normalization_y = np.sqrt(_debiased_dot_product_similarity_helper(
            normalization_y ** 2, sum_squared_rows_y, sum_squared_rows_y,
            squared_norm_y, squared_norm_y, n))

    return dot_product_similarity / (normalization_x * normalization_y)


# In[ ]:


# change folder to the netfound src folder

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

def load_data(path, batch_size = 1):
    tdataset = load_dataset("arrow", data_dir=path, split="train", cache_dir="/dev/shm/cachetmp", streaming=False)
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

def load_model(path: str):
    config = NetFoundLarge()
    return NetfoundFinetuningModel.from_pretrained(path, config=config, ignore_mismatched_sizes=True).to("cpu")


# In[3]:


BATCH_SIZE=512
LIMIT=50000


# In[ ]:


# load model and dataset
model = load_model("/global/homes/k/kell/scratch/ucsb_big_data/pretraining/jan17-48hrs-80gb/checkpoint-240000")
model = model.to("cuda")

_, dataloader = load_data("/dev/shm/data/netfound", batch_size=BATCH_SIZE)


# In[5]:


def get_embedding(dataset, model, _flow_mask, random=False):
    "takes dataset, model, and binary mask for indices allowed for perturbation, generates random noise tensor with respect to the mask, applies it to the dataset, gets embeddings, returns resulting noise tensor and embeddings"
    embeddings = []
    noises = []
    sims = []
    total = 0
    for batch in tqdm(dataset):
        orig_input = batch['input_ids'].clone().cpu()
        if not random:
            noise = batch['input_ids'].clone().cpu()
            for col in range(batch['input_ids'].size(1)):
                perm = torch.randperm(batch['input_ids'].size(0))
                noise[:, col] = batch['input_ids'][perm, col]
        else:
            noise = torch.randint(low=0, high=65535, size=batch['input_ids'].size(), device=batch['input_ids'].device)  # random noise in the whole range of available tokens

        flow_mask = _flow_mask.repeat(batch['input_ids'].size(0), 1).to(batch['input_ids'].device)
        flow_mask = flow_mask[:, :batch['input_ids'].size(1)].bool()
        batch['input_ids'][flow_mask] = noise[flow_mask]
        sims.append((batch['input_ids'] == orig_input).float().mean().item())
        perturbed_features, _ = encode(batch, model)
        embeddings.append(perturbed_features)
        total += batch['input_ids'].size(0)
        if total > LIMIT:
            break

    print(f"Similarity: {np.mean(sims)}")
    embeddings = torch.cat(embeddings)
    return embeddings  


# In[6]:


# packet mask for tcp: 1 CLS token + (12 TCP tokens + 6 PAYLOAD)
# full flow mask: (packet_mask * 6 packets/burst) * 12 bursts

def repeat_mask(packet_mask):
    "takes packet mask and turns it into full-length netfound mask (packet, burst, flow)"
    burst_mask = packet_mask.repeat(1, 6)
    burst_mask = torch.cat((torch.tensor([0]).unsqueeze(0), burst_mask), dim=1)
    flow_mask = burst_mask.repeat(1, 12)
    return flow_mask


# In[7]:


def _perturb(packet_mask, random=False):
    return get_embedding(dataloader, model, repeat_mask(packet_mask), random=random)


# In[8]:


from sklearn.feature_selection import mutual_info_regression
import torch

# define correlation calculation function

# def calculate_correlation(emb_original: torch.Tensor, noise: torch.Tensor, emb_perturbed: torch.Tensor) -> (float, float, np.ndarray):
#     '''accepts original embedding, noise, and embedding after perturbation, and calculates similarity correlation between noise and each dimension of the perturbation result'''
#     emb_diff = emb_perturbed - emb_original

#     cos_sim = torch.nn.functional.cosine_similarity(emb_perturbed, emb_original).mean()
#     l2_dist = torch.cdist(emb_perturbed, emb_original, p=2).mean()

#     emb_diff_np = emb_diff.detach().cpu().numpy()
#     noise_np = noise.detach().cpu().numpy()
#     noise_np = noise_np[:, noise_np.any(axis=0)]  # keep only non zero noise columns effectively removing masked out columns 

#     n_dims = emb_diff_np.shape[1]
#     cka_scores = np.zeros(n_dims)
    
#     for d in tqdm(range(n_dims)):
#         # extract the nth column as a 2D array
#         feature_column = emb_diff_np[:, d].reshape(-1, 1)
#         cka_scores[d] = feature_space_linear_cka(noise_np, feature_column)

#     return cos_sim, l2_dist, cka_scores

def calculate_correlation(emb_original: torch.Tensor, emb_perturbed: torch.Tensor):
    # shorter version - only cos sim
    return torch.nn.functional.cosine_similarity(emb_perturbed, emb_original).mean()


# In[9]:


# original embeddings
mask = torch.tensor([0] * 12 + [0] * 6)  # zero mask
original_embeddings = _perturb(mask)


# In[10]:


def calculate_similarity(mask):
    with torch.no_grad():
        new_emb = _perturb(mask, random=True)
        cos_sim = calculate_correlation(original_embeddings, new_emb)
        print(f"Cos sim for random source perturbation: {cos_sim}")
    
        new_emb = _perturb(mask)
        cos_sim = calculate_correlation(original_embeddings, new_emb)
        print(f"Cos sim for reordered perturbation: {cos_sim}")


# In[11]:


# payload
calculate_similarity(torch.tensor([0] * 12 + [1] * 6))


# In[12]:


# SEQ/ACK
calculate_similarity(torch.tensor([0] * 7 + [1] * 4 + [0] * 7))


# In[13]:


# IP total length
calculate_similarity(torch.tensor([0] * 2 + [1] * 1 + [0] * 15))


# In[14]:


# IP TTL
calculate_similarity(torch.tensor([0] * 4 + [1] * 1 + [0] * 13))


# In[15]:


# TCP Flags
calculate_similarity(torch.tensor([0] * 5 + [1] * 1 + [0] * 12))


# In[16]:


# TCP WSize
calculate_similarity(torch.tensor([0] * 6 + [1] * 1 + [0] * 11))


# In[ ]:




