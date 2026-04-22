#!/usr/bin/env python
# coding: utf-8

# # netmamba feature perturbation

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


# change folder to netmamba src

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

os.environ['PATH'] = '/sbin:' + os.environ.get('PATH', '')


# In[ ]:


def get_loader(data_path, batch_size=1):
    mean = [0.5]
    std = [0.5]

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    dataset = datasets.ImageFolder(data_path, transform=transform)
    sampler = torch.utils.data.SequentialSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, sampler=sampler,
        batch_size=batch_size,
        num_workers=1
    )
    return dataloader

def get_model():
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
    model = model.to("cuda")
    return model


# In[5]:


BATCH_SIZE = 4096
LIMIT = 300000


# In[ ]:


dataloader = get_loader("../data/netmamba/array_sampled", batch_size=BATCH_SIZE)
model = get_model()


# In[7]:


"""
data layout is the same as YaTC
header in YaTC is 80 floats:
48:56 are seq
56:64 are ack
6:8 is total length
16:18 is TTL
flags are 12, 20, 23
68:76 is WSize

then we have 240 floats of payload
total 320 floats for a single packet
total 5 packet = 1600 floats
reshaped to 40, 40 getting a single image
"""


# In[8]:


def repeat_mask(packet_mask: np.ndarray):
    "320 floats to 1600 and reshape"
    assert packet_mask.shape == (320,)
    return np.reshape(np.tile(packet_mask, 5), (40, 40))


# In[9]:


def encode(batch, model):
    batch = batch.to("cuda")
    with torch.no_grad():
        return model.forward_encoder(batch, mask_ratio=0.0, if_mask=False)[:, -1, :].cpu()


# In[10]:


def get_embedding(dataset, model, packet_mask, random=False):
    "takes dataset, model, and binary mask for indices allowed for perturbation, generates random noise tensor with respect to the mask, applies it to the dataset, gets embeddings, returns resulting noise tensor and embeddings"
    embeddings = []
    noises = []
    total = 0
    sims = []
    mask_np = repeat_mask(packet_mask)
    for batch, _ in tqdm(dataset):
        orig_input = batch.clone()
        B, C, H, W = batch.size()
        noise = torch.empty_like(batch)

        if not random:
            for h in range(H):
                for w in range(W):
                    perm = torch.randperm(B)
                    noise[:, :, h, w] = batch[perm, :, h, w]
        else:
            noise = torch.rand((batch.size(0), 1, 40, 40)) * 2 - 1  # from -1 to +1

        mask_torch = torch.from_numpy(mask_np).bool().unsqueeze(0).unsqueeze(0).cpu()  # (1, 1, 40, 40)
        mask_torch = mask_torch.repeat(batch.size(0), 1, 1, 1)
        batch[mask_torch] = noise[mask_torch]
        perturbed_features = encode(batch, model)
        embeddings.append(perturbed_features)
        sims.append((batch == orig_input).float().mean().item())
        total += batch.size(0)
        if total > LIMIT:
            break
        
    print(f"Similarity: {np.mean(sims)}")
    embeddings = torch.cat(embeddings)
    return embeddings    


# In[11]:


def _perturb(packet_mask, random=False):
    return get_embedding(dataloader, model, packet_mask, random)


# In[12]:


from sklearn.feature_selection import mutual_info_regression
import torch

# define correlation calculation function

# def calculate_correlation(emb_original: torch.Tensor, noise: torch.Tensor, emb_perturbed: torch.Tensor) -> np.ndarray:
#     '''accepts original embedding, noise, and embedding after perturbation, and calculates similarity correlation between noise and each dimension of the perturbation result'''
#     emb_diff = emb_perturbed - emb_original

#     cos_sim = torch.nn.functional.cosine_similarity(emb_perturbed, emb_original).mean()
#     l2_dist = torch.cdist(emb_perturbed, emb_original, p=2).mean()

#     emb_diff_np = emb_diff.detach().cpu().numpy()
#     noise_np = noise.detach().cpu().numpy()
#     noise_np = np.reshape(noise_np, (noise_np.shape[0], -1))
#     noise_np = noise_np[:, noise_np.any(axis=0)]  # keep only non zero noise columns effectively removing masked out columns 

#     n_dims = emb_diff_np.shape[1]
#     cka_scores = np.zeros(n_dims)
    
#     for d in tqdm(range(n_dims)):
#         # extract the nth column as a 2D array
#         feature_column = emb_diff_np[:, d].reshape(-1, 1)
#         cka_scores[d] = feature_space_linear_cka(noise_np, feature_column)

#     return cos_sim, l2_dist, cka_scores

def calculate_correlation(emb_original: torch.Tensor, emb_perturbed: torch.Tensor) -> float:
    # simplified
    return torch.nn.functional.cosine_similarity(emb_perturbed, emb_original).mean()


# In[13]:


# original embeddings
mask = np.array([0] * 320)  # zero mask
original_embeddings = _perturb(mask)


# In[ ]:


original_embeddings.shape


# In[ ]:


def calculate_similarity(mask):
    new_emb, new_noise = _perturb(mask)
    cos_sim, l2_dist, correlation = calculate_correlation(original_embeddings, new_noise, new_emb)
    top5_sim = np.argsort(correlation)[-5:]
    print(f"Cos sim: {cos_sim}")
    print(f"L2 distance: {l2_dist}")
    print(f"Average similarity: {np.mean(correlation)}")
    print(f"Top 5 indices: {top5_sim[::-1]}")
    print(f"Top 5 similarity values: {correlation[top5_sim][::-1]}")


# In[ ]:


# simplified
def calculate_similarity(mask):
    new_emb = _perturb(mask, random=True)
    cos_sim = calculate_correlation(original_embeddings, new_emb)
    print(f"Cos sim for random source perturbation: {cos_sim}")

    new_emb = _perturb(mask)
    cos_sim = calculate_correlation(original_embeddings, new_emb)
    print(f"Cos sim for reordered perturbation: {cos_sim}")


# In[ ]:


# payload
calculate_similarity(np.array([0] * 80 + [1] * 240))


# In[ ]:


# SEQ/ACK
# 48:56 are seq
# 56:64 are ack
calculate_similarity(np.array([0] * 48 + [1] * 16 + [0] * 16 + [0] * 240))


# In[ ]:


# IP total length
# 6:8 is total length
calculate_similarity(np.array([0] * 6 + [1] * 2 + [0] * (72 + 240)))


# In[ ]:


# IP TTL
# 16:18 is TTL
calculate_similarity(np.array([0] * 16 + [1] * 2 + [0] * (62 + 240)))


# In[ ]:


# TCP Flags
# flags are 12, 20, 23
calculate_similarity(np.array(
    [0] * 12 + [1] * 1 + 
    [0] * 7 + [1] * 1 + 
    [0] * 2 + [1] * 1 +
    [0] * (56 + 240)
))


# In[ ]:


# TCP WSize (68:76)
calculate_similarity(np.array([0] * 68 + [1] * 8 + [0] * (4 + 240)))


# In[ ]:




