#!/usr/bin/env python
# coding: utf-8

# # etbert feature perturbation

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


# change folder to the source of the etbert

from finetuning.run_classifier import Classifier, read_dataset, load_or_initialize_parameters, count_labels_num, batch_loader
import torch
import numpy as np
from collections import defaultdict
from argparse import Namespace
from uer.layers import *
from uer.encoders import *
from uer.utils.constants import *
from uer.utils import *
from uer.utils.optimizers import *
from uer.opts import finetune_opts
from uer.utils.config import load_hyperparam
import argparse
from tqdm import tqdm
import copy
import threading
import pickle


# In[ ]:


def get_data_model(datafolder, batch_size=1, pretrained_model="../models/ET-BERT/pretrained_model_etbert.bin"):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    finetune_opts(parser)
    args = []
    args += ["--train_path", "dummy"]
    args += ["--vocab_path", "../models/ET-BERT/src/models/encryptd_vocab.txt"]
    args += ["--dev_path", "../models/ET-BERT/cic/test_dataset.tsv"]
    args += ["--pretrained_model_path", "dummy"]
    args = parser.parse_args(args)
    args.tokenizer = "bert"
    args.pooling = "first"
    args.soft_targets = False
    args.topk = 1
    args.frozen = False
    args.soft_alpha = 0.5
    
    args = load_hyperparam(args)
    args.tokenizer = str2tokenizer[args.tokenizer](args)
    args.batch_size = batch_size
    args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    def get_model(args, model_path, train_path):
        args.pretrained_model_path = model_path
        args.train_path = train_path
        args.labels_num = count_labels_num(args.train_path)
        model = Classifier(args)
        load_or_initialize_parameters(args, model)
        model = model.to(args.device)
        return model

    etbert_frozen_model = get_model(args,pretrained_model, f"{datafolder}/train_dataset.tsv")
    data_etbert = read_dataset(args, f"{datafolder}/train_dataset.tsv")

    return data_etbert, etbert_frozen_model


# In[ ]:


data, model = get_data_model("../data/etbert/output", batch_size=BATCH_SIZE, pretrained_model="../models/ET-BERT/pretrained_model_etbert.bin")


# In[8]:


def encode(model, src_batch, seg_batch):
    with torch.no_grad():
        src_batch = src_batch.to("cuda:1")
        seg_batch = seg_batch.to("cuda:1")
        emb = model.embedding(src_batch, seg_batch)
        emb = model.encoder(emb, seg_batch)
        emb = emb[:, 0, :]   # pooling = first
    return emb.cpu()


# In[9]:


LIMIT=300000


# In[10]:


def get_embedding(dataset, model, random_mask, batch_size=BATCH_SIZE, random=False):
    "takes dataset, model, and binary mask for indices allowed for perturbation, generates random noise tensor with respect to the mask, applies it to the dataset, gets embeddings, returns resulting noise tensor and embeddings"
    assert random_mask.shape == (128,)
    print(f"random mask density: {random_mask.float().mean().item() * 100}")
    embeddings = []
    total = 0
    src = torch.LongTensor([example[0] for example in dataset])

    # noise is a random permutation of src in each column or random noise completely
    old_src = src.clone()
    if not random:
        noise = src.clone()
        for col in range(128):
            perm = torch.randperm(src.size(0))
            noise[:, col] = src[perm, col]
    else:
        noise = torch.randint(low=6, high=60004, size=src.size(), device=src.device)  # random noise in the whole range of available tokens

    assert random_mask.shape == (128,)
    random_mask = random_mask.unsqueeze(0)
    random_mask = random_mask.repeat(src.size(0), 1).bool()
    src[random_mask] = noise[random_mask]
    print(f"Similarity: {(src == old_src).float().mean().item()}")

    tgt = torch.LongTensor([example[1] for example in dataset])
    seg = torch.LongTensor([example[2] for example in dataset])
    loader = batch_loader(batch_size, src, tgt, seg, None)
    for src, tgt, seg, _ in tqdm(loader):
        perturbed_features = encode(model, src, seg)
        embeddings.append(perturbed_features)
        total += src.size(0)
        if total > LIMIT:
            break

    embeddings = torch.cat(embeddings)
    return embeddings


# In[11]:


def _perturb(random_mask, random=False):
    return get_embedding(data, model, random_mask, random=random)


# In[12]:


from sklearn.feature_selection import mutual_info_regression
import torch

# define correlation calculation function

def calculate_correlation(emb_original: torch.Tensor, noise: torch.Tensor, emb_perturbed: torch.Tensor) -> np.ndarray:
    '''accepts original embedding, noise, and embedding after perturbation, and calculates similarity correlation between noise and each dimension of the perturbation result'''
    emb_diff = emb_perturbed - emb_original

    cos_sim = torch.nn.functional.cosine_similarity(emb_perturbed, emb_original).mean()
    l2_dist = torch.cdist(emb_perturbed, emb_original, p=2).mean()

    emb_diff_np = emb_diff.detach().cpu().numpy()
    noise_np = noise.detach().cpu().numpy()
    noise_np = np.reshape(noise_np, (noise_np.shape[0], -1))
    noise_np = noise_np[:, noise_np.any(axis=0)]  # keep only non zero noise columns effectively removing masked out columns 

    n_dims = emb_diff_np.shape[1]
    cka_scores = np.zeros(n_dims)
    
    for d in tqdm(range(n_dims)):
        # extract the nth column as a 2D array
        feature_column = emb_diff_np[:, d].reshape(-1, 1)
        cka_scores[d] = feature_space_linear_cka(noise_np, feature_column)

    return cos_sim, l2_dist, cka_scores

def calculate_correlation(emb_original: torch.Tensor,  emb_perturbed: torch.Tensor):
    # simplified
    return torch.nn.functional.cosine_similarity(emb_perturbed, emb_original).mean()


# In[13]:


# original embeddings
original_embeddings = _perturb(torch.tensor([0] * 128))


# In[14]:


# def calculate_similarity(mask):
#     new_emb, new_noise = _perturb(mask)
#     cos_sim, l2_dist, correlation = calculate_correlation(original_embeddings, new_noise, new_emb)
#     top5_sim = np.argsort(correlation)[-5:]
#     print(f"Cos sim: {cos_sim}")
#     print(f"L2 distance: {l2_dist}")
#     print(f"Average similarity: {np.mean(correlation)}")
#     print(f"Top 5 indices: {top5_sim[::-1]}")
#     print(f"Top 5 similarity values: {correlation[top5_sim][::-1]}")
#     return cos_sim

# simplified
def calculate_similarity(mask):
    new_emb = _perturb(mask)
    cos_sim = calculate_correlation(original_embeddings, new_emb)
    print(f"Cos sim for reordered perturbation: {cos_sim}")
    
    new_emb = _perturb(mask, random=True)
    cos_sim = calculate_correlation(original_embeddings, new_emb)
    print(f"Cos sim for random source perturbation: {cos_sim}")


# In[15]:


# 100% random payload
calculate_similarity(torch.tensor([1] * 128))


# In[16]:


# first 25%
calculate_similarity(torch.tensor([1] * 32 + [0] * 96))


# In[17]:


# second 25%
calculate_similarity(torch.tensor([0] * 32 + [1] * 32 + [0] * 64))


# In[18]:


# third 25%
calculate_similarity(torch.tensor([0] * 64 + [1] * 32 + [0] * 32))


# In[19]:


# last 25%
calculate_similarity(torch.tensor([0] * 96 + [1] * 32))


# In[20]:


# 50%
calculate_similarity(torch.tensor([1] * 64 + [0] * 64))


# In[21]:


# 50%
calculate_similarity(torch.tensor([0] * 64 + [1] * 64))


# In[ ]:




