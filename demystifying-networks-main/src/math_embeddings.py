#!/usr/bin/env python
# coding: utf-8

# ## Embedding math

# ## perf dataset: iperf vs qperf, cubic vs reno

# In[1]:


data = {
    "etbert": "../data/etbert/perf_emb.pkl",
    "yatc": "../data/yatc/perf_emb.pkl",
    "netmamba": "../data/netmamba/netmamba_perf_emb.pkl",
    "netfound": "../data/netfound/perf_emb.pkl",
}

mapping = lambda x: x.split("/")[-1].split('.')[0][:-5]


# In[2]:


import torch
import torch.nn.functional as F
import pickle
import os
import numpy as np

def measure_spread(tensor1, tensor2):
    similarity_matrix = F.cosine_similarity(tensor1.unsqueeze(1), tensor2.unsqueeze(0), dim=2)
    return similarity_matrix.mean()

def measure_dist(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    diff = torch.abs(tensor1.unsqueeze(1) - tensor2.unsqueeze(0))
    distances = diff.sum(dim=2)
    return distances.mean().item()


# In[3]:


for model, path in data.items():
    with open(path, "rb") as f:
        embeddings, labels = pickle.load(f)
    labels = [mapping(x) for x in labels]
    print(labels)


# In[4]:


"""
experiments:
'qperf-cubic' - QUIC + Cubic
'qperf-reno' - QUIC + Reno
'iperf-cubic' - TCP + Cubic
'iperf-reno' - TCP + Reno
"""

order = ['qperf-cubic', 'iperf-reno', 'iperf-cubic', 'qperf-reno']

for model, path in data.items():
    with open(path, "rb") as f:
        embeddings, labels = pickle.load(f)
    labels = [mapping(x) for x in labels]
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.size(0) == len(labels)
    classes = {
        x: embeddings[torch.tensor([label == x for label in labels], dtype=torch.bool)]
        for x in set(labels)
    }

    mean_cos_sim = np.mean([measure_spread(classes[label], classes[label]) for label in classes])
    mean_l1_dist = np.mean([measure_dist(classes[label], classes[label]) for label in classes])

    # self-similarity for baseline - stability test
    print(f"Model {model}, stability: avg_cosine_similarity = {mean_cos_sim:.4f}, avg_L1_dist = {mean_l1_dist:.0f}")

    import itertools
    pairs = list(itertools.combinations(order, 2))
    for pair in pairs:
        print(f"Model {model}, distance between {pair[0]} and {pair[1]}: avg_cosine_similarity = {measure_spread(classes[pair[0]], classes[pair[1]]):.4f}, avg_L1_dist = {measure_dist(classes[pair[0]], classes[pair[1]]):.0f}")

    # math
    resulting_emb = classes[order[0]] + classes[order[1]] - classes[order[2]]
    for i in range(4):
        if i == 3:
            print("Target ", end='')
        print(f"Model {model}, distance from ({order[0]} + {order[1]} - {order[2]}) to {order[i]}: avg_cosine_similarity = {measure_spread(resulting_emb, classes[order[i]]):.4f}, avg_L1_dist = {measure_dist(resulting_emb, classes[order[i]]):.0f}")
    print()
        
        


# In[5]:


"""
experiments:
'qperf-cubic' - QUIC + Cubic
'qperf-reno' - QUIC + Reno
'iperf-cubic' - TCP + Cubic
'iperf-reno' - TCP + Reno
"""

order = ['qperf-cubic', 'iperf-reno', 'iperf-cubic', 'qperf-reno']

def intra_class_spread(t):
    n = t.size(0)
    sims = F.cosine_similarity(t.unsqueeze(1), t.unsqueeze(0), dim=2)
    off_diag = sims[~torch.eye(n, dtype=torch.bool, device=t.device)]
    return off_diag.mean().item()

for model, path in data.items():
    with open(path, "rb") as f:
        embeddings, labels = pickle.load(f)
    labels = [mapping(x) for x in labels]
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.size(0) == len(labels)
    classes = {
        x: embeddings[torch.tensor([label == x for label in labels], dtype=torch.bool)]
        for x in set(labels)
    }

    mean_cos_sim = np.mean([intra_class_spread(classes[label]) for label in classes])
    mean_l1_dist = np.mean([measure_dist(classes[label], classes[label]) for label in classes])

    # self-similarity for baseline - stability test
    print(f"Model {model}, stability: avg_cosine_similarity = {mean_cos_sim:.4f}, avg_L1_dist = {mean_l1_dist:.0f}")

    import itertools
    pairs = list(itertools.combinations(order, 2))
    for pair in pairs:
        centroids = {k: v.mean(dim=0) for k, v in classes.items()}
        result_vec = centroids['qperf-cubic'] + centroids['iperf-reno'] - centroids['iperf-cubic']

        print(f"Model {model}, distance between {pair[0]} and {pair[1]}: avg_cosine_similarity = {measure_spread(classes[pair[0]], classes[pair[1]]):.4f}, avg_L1_dist = {measure_dist(classes[pair[0]], classes[pair[1]]):.0f}")

    # math
    resulting_emb = (classes[order[0]].mean(dim=0) + classes[order[1]].mean(dim=0) - classes[order[2]].mean(dim=0)).unsqueeze(0)
    for i in range(4):
        if i == 3:
            print("Target ", end='')
        print(f"Model {model}, distance from ({order[0]} + {order[1]} - {order[2]}) to {order[i]}: avg_cosine_similarity = {measure_spread(resulting_emb, classes[order[i]]):.4f}, avg_L1_dist = {measure_dist(resulting_emb, classes[order[i]]):.0f}")
    print()
        
        


# In[6]:


import torch, pickle, itertools
import torch.nn.functional as F
from pathlib import Path

data = {
    "etbert": "../data/etbert/perf_emb.pkl",
    "yatc":   "../data/yatc/perf_emb.pkl",
    "netmamba": "../data/netmamba/netmamba_perf_emb.pkl",
    "netfound": "../data/netfound/perf_emb.pkl",
}

order = ["qperf-cubic", "iperf-reno", "iperf-cubic", "qperf-reno"]
    

def pairwise_cos(a, b):
    return (F.normalize(a, dim=1) @ F.normalize(b, dim=1).T).mean().item()

def l1_mean(a, b):
    return torch.cdist(a, b, p=1).mean().item()

for model, path in data.items():
    emb, raw_labels = pickle.load(open(path, "rb"))
    labels = [mapping(x) for x in raw_labels]

    cls = {lab: emb[torch.tensor([l == lab for l in labels])] for lab in set(labels)}

    # intra-class spread without self-pairs
    intra = {k: pairwise_cos(v, v) for k, v in cls.items()}
    print(f"{model}: intra-class cosine (no diagonal) {intra}")

    # pairwise class similarity
    for a, b in itertools.combinations(order, 2):
        print(f"{model}: cos({a},{b}) = {pairwise_cos(cls[a], cls[b]):.4f}, "
              f"L1 = {l1_mean(cls[a], cls[b]):.2f}")

    # centroids & analogy
    centroids = {k: v.mean(0) for k, v in cls.items()}
    analog_vec = centroids["qperf-cubic"] + centroids["iperf-reno"] - centroids["iperf-cubic"]

    for tgt in order:
        cos = F.cosine_similarity(analog_vec, centroids[tgt], dim=0).item()
        l1  = torch.dist(analog_vec, centroids[tgt], p=1).item()
        tag = "(target)" if tgt == "qperf-reno" else ""
        print(f"{model}: ‖qC + iR − iC – {tgt}‖  cos={cos:.4f}  L1={l1:.2f} {tag}")

    # delta-vectors check
    print(f"TCP/QUIC: {model}: analogy delta cosine = {F.cosine_similarity(centroids['qperf-cubic'] - centroids['iperf-cubic'], centroids['qperf-reno']  - centroids['iperf-reno'], dim=0).item():.4f}")
    print(f"Cubic/Reno: {model}: analogy delta cosine = {F.cosine_similarity(centroids['qperf-reno'] - centroids['qperf-cubic'], centroids['iperf-reno'] - centroids['iperf-cubic'], dim=0).item():.4f}\n")


# ## synth dataset: aqm, cc, cross

# In[7]:


import torch
import torch.nn.functional as F
import pickle
import os
import numpy as np

def measure_spread(A, B, exclude_self=False):
    M = F.cosine_similarity(A.unsqueeze(1), B.unsqueeze(0), dim=2)
    if exclude_self and A.data_ptr()==B.data_ptr():
        M = M[~torch.eye(len(A), dtype=torch.bool, device=A.device)]
    return M.mean()

def measure_dist(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    diff = torch.abs(tensor1.unsqueeze(1) - tensor2.unsqueeze(0))
    distances = diff.sum(dim=2)
    return distances.mean().item()

data = {
    "etbert": "../data/etbert/synth_emb.pkl",
    "yatc": "../data/yatc/synth_emb.pkl",
    "netmamba": "../data/netmamba/netmamba_synth_emb.pkl",
    "netfound": "../data/netfound/synth_emb.pkl",
}

mapping = lambda x: '_'.join(x.split("/")[-1].split('.')[0].split("_")[:-1])

"""
experiments:
'fifo_6m_bbr_prof50_36' - baseline
'fifo_6m_cubic_prof50_36' - different cc algorithm
'codel_6m_bbr_prof50_36' - different AQM
'fifo_6m_bbr_prof72_29' - different cross traffic
'codel_6m_cubic_prof72_29' - different cc, aqm, and cross traffic

idea: cc + aqm + cross - 2*baseline == all
should work because each of 3 single change vectors have 1 change and 2 variables the same
fifo_6m_cubic_prof50_36 + codel_6m_bbr_prof50_36 + fifo_6m_bbr_prof72_29 = 2x fifo, 2x bbr, 2x prof50, 1x cubic, 1x codel, 1x prof72
so we retract 2x fifo_6m_bbr_prof50_36 and should get 1x cubic, 1x codel, 1x prof72 which is codel_6m_cubic_prof72_29
"""

order = ['fifo_6m_cubic_prof50_36', 'codel_6m_bbr_prof50_36', 'fifo_6m_bbr_prof72_29', 'fifo_6m_bbr_prof50_36', 'codel_6m_cubic_prof72_29']

for model, path in data.items():
    with open(path, "rb") as f:
        embeddings, labels = pickle.load(f)
    labels = [mapping(x) for x in labels]
    assert isinstance(embeddings, torch.Tensor)
    assert embeddings.size(0) == len(labels)
    classes = {
        x: embeddings[torch.tensor([label == x for label in labels], dtype=torch.bool)]
        for x in set(labels)
    }

    mean_cos_sim = np.mean([measure_spread(classes[label], classes[label], exclude_self=True) for label in classes])
    mean_l1_dist = np.mean([measure_dist(classes[label], classes[label]) for label in classes])

    # self-similarity for baseline - stability test
    print(f"Model {model}, stability: avg_cosine_similarity = {mean_cos_sim:.4f}, avg_L1_dist = {mean_l1_dist:.1f}")
    print(f"Model {model}, baseline stability: avg_cosine_similarity = {measure_spread(classes['fifo_6m_bbr_prof50_36'], classes['fifo_6m_bbr_prof50_36'], exclude_self=True):.4f}, avg_L1_dist = {measure_dist(classes['fifo_6m_bbr_prof50_36'], classes['fifo_6m_bbr_prof50_36']):.1f}")

    import itertools
    pairs = list(itertools.combinations(order, 2))
    for pair in pairs:
        print(f"Model {model}, distance between {pair[0]} and {pair[1]}: avg_cosine_similarity = {measure_spread(classes[pair[0]], classes[pair[1]]):.4f}, avg_L1_dist = {measure_dist(classes[pair[0]], classes[pair[1]]):.1f}")

    # math
    # getting centroids
    resulting_emb = (classes[order[0]].mean(dim=0) + classes[order[1]].mean(dim=0) + classes[order[2]].mean(dim=0) - 2*classes[order[3]].mean(dim=0)).unsqueeze(0)
    for i in range(5):
        if i == 4:
            print("Target ", end='')
        print(f"Model {model}, distance from ({order[0]} + {order[1]} + {order[2]} - 2x {order[3]}) to {order[i]}: avg_cosine_similarity = {measure_spread(resulting_emb, classes[order[i]]):.4f}, avg_L1_dist = {measure_dist(resulting_emb, classes[order[i]]):.1f}")
    print()
        
        


# In[ ]:




