#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.metrics import silhouette_score
import pickle
import torch
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm


# In[2]:


def train(embeddings, labels, classes, epochs=30, test_split=0.2, normalize=False):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm import tqdm
    import matplotlib.pyplot as plt

    # Use CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Shuffle data
    indices = torch.randperm(embeddings.size(0))
    shuffled_data = embeddings[indices].to(device)
    shuffled_labels = [labels[i] for i in indices.tolist()]
    
    # Map labels to indices
    unique_labels = sorted(set(shuffled_labels))
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    y = torch.tensor([label_to_idx[label] for label in shuffled_labels], dtype=torch.long).to(device)
    
    # Split data into training and test sets
    total_samples = shuffled_data.size(0)
    train_size = int((1 - test_split) * total_samples)
    
    train_data = shuffled_data[:train_size]
    train_labels = y[:train_size]
    test_data = shuffled_data[train_size:]
    test_labels = y[train_size:]
    
    # Prepare DataLoader for training data
    train_dataset = TensorDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Define model and send to device
    # it was linear128 - relu - dropout - linear before
    layers = [
        nn.Linear(embeddings.size(1), classes),
    ]
    if normalize:
        layers.insert(0, nn.BatchNorm1d(embeddings.size(1)))
    model = nn.Sequential(*layers).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    epoch_losses = []
    model.train()
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)
        epoch_losses.append(avg_loss)
    
    # Plot training loss
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs + 1), epoch_losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Over Epochs")
    plt.grid(True)
    plt.show()
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        outputs = model(train_data)
        _, preds = torch.max(outputs, 1)
    
    acc = accuracy_score(train_labels.cpu().numpy(), preds.cpu().numpy())
    f1 = f1_score(train_labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
    print(f"Train Accuracy: {acc:.4f}, Train F1 Score: {f1:.4f}")
    
    with torch.no_grad():
        outputs = model(test_data)
        _, preds = torch.max(outputs, 1)
    
    acc = accuracy_score(test_labels.cpu().numpy(), preds.cpu().numpy())
    f1 = f1_score(test_labels.cpu().numpy(), preds.cpu().numpy(), average='weighted')
    print(f"Test Accuracy: {acc:.4f}, Test F1 Score: {f1:.4f}")
    
    return f1



# In[3]:


# from WhiteningBERT
def whitening_torch_final(embeddings):
    mu = torch.mean(embeddings, dim=0, keepdim=True)
    cov = torch.mm((embeddings - mu).t(), embeddings - mu)
    u, s, vt = torch.svd(cov.float())
    W = torch.mm(u, torch.diag(1/torch.sqrt(s)))
    embeddings = torch.mm(embeddings - mu, W)
    return embeddings


# In[4]:


def decorr_only(embeddings):
    cov = embeddings.T @ embeddings / (embeddings.size(0) - 1)
    u, _, _ = torch.svd(cov.float())
    embeddings_decorrelated = embeddings @ u
    return embeddings_decorrelated


# In[5]:


def f1_improvement(filepath, classes, mapping, epochs=30):
    with open(filepath, "rb") as f:
        embeddings, filenames = pickle.load(f)

    filenames = [mapping(x) for x in filenames]
    f1 = train(embeddings, filenames, classes, epochs)
    f1_d = train(decorr_only(embeddings), filenames, classes, epochs)
    f1_n = train(embeddings, filenames, classes, epochs, normalize=True)
    f1_w = train(whitening_torch_final(embeddings), filenames, classes, epochs)
    print(f"Decorrelation improvement: {f1_d - f1}")
    print(f"BatchNorm improvement: {f1_n - f1}")
    print(f"Whitening improvement: {f1_w - f1}")


# ## netmamba

# In[6]:


def mapping(x):
    return x.removeprefix("/dev/shm/data/").split('/')[2]


# In[7]:


# netmamba - Crossmarket
f1_improvement("../data/netmamba/netmamba_cross_emb.pkl", 210, mapping)


# In[8]:


# netmamba - cicids17
f1_improvement("../data/netmamba/netmamba_cicids_emb.pkl", 8, mapping)


# In[ ]:


# netmamba - cicapt
f1_improvement("../data/netmamba/netmamba_cicapt_emb.pkl", 22, mapping)


# ## yatc

# In[ ]:


def mapping(x):
    return x.removeprefix("/dev/shm/data/").split('/')[2]


# In[ ]:


# yatc - Crossmarket
f1_improvement("../data/yatc/cross_emb.pkl", 210, mapping)


# In[ ]:


# yatc - cicids17
f1_improvement("../data/yatc/cicids_emb.pkl", 8, mapping)


# In[ ]:


# yatc - cicapt
f1_improvement("../data/yatc/cicapt_emb.pkl", 22, mapping)


# ## etbert

# In[ ]:


def mapping(x):
    return [i for i in x.removeprefix("/dev/shm/data/").removeprefix("/dev/shm/data2/").split('/') if i != ''][1]


# In[ ]:


# etbert - Crossmarket
f1_improvement("../data/etbert/cross_emb.pkl", 210, mapping)


# In[ ]:


# etbert - cicids17
f1_improvement("../data/etbert/cicids_emb.pkl", 8, mapping)


# In[ ]:


# etbert - cicapt
f1_improvement("../data/etbert/cicapt_emb.pkl", 22, mapping)


# ## netfound

# In[ ]:


# netfound - Crossmarket
def mapping(x):
    return x.removeprefix("/data/").split('-')[0]

f1_improvement("../data/netfound/cross_emb.pkl", 210, mapping)


# In[ ]:


# netfound - cicids17
def mapping(x):
    return '-'.join(x.removeprefix('/data/').split('_')[1:3]).split('.')[0]

f1_improvement("../data/netfound/cicids_emb.pkl", 8, mapping)


# In[ ]:


# netfound - cicapt
def mapping(x):
    return '-'.join(x.removeprefix("/data/").split('-')[:3])

f1_improvement("../data/netfound/cicapt_emb.pkl", 22, mapping)


# In[ ]:


# custom calculations

# netfound - Crossmarket
def mapping(x):
    return x.removeprefix("/data/").split('-')[0]

with open("../data/netfound/cross_emb.pkl", "rb") as f:
    embeddings, filenames = pickle.load(f)

filenames = [mapping(x) for x in filenames]
f12 = train(whitening_torch_final(embeddings), filenames, 210, 60)
print(f12)


# In[ ]:




