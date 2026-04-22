#!/usr/bin/env python
# coding: utf-8

# ## YaTC embeddings calculation

# In[ ]:


get_ipython().run_line_magic('cd', '../models/YaTC/src')


# In[2]:


import pickle


# In[3]:


from tqdm.auto import tqdm
import copy
import threading

from finetune import build_dataset
import torch
import models_YaTC
from util.pos_embed import interpolate_pos_embed
from tqdm import tqdm
from collections import defaultdict


# In[ ]:


def get_embeddings(datafolder, n_classes, batch_size=64, limit = 10**30, gpus=4):
    loader_yatc = lambda: True
    loader_yatc.data_path = datafolder
    loader_yatc = build_dataset(is_train=True, args=loader_yatc)
    loader_yatc = torch.utils.data.DataLoader(
            loader_yatc, sampler=torch.utils.data.SequentialSampler(loader_yatc),
            batch_size=batch_size,
            num_workers=4,
            pin_memory=False,
            drop_last=True,
        )

    checkpoint_model = torch.load("../models/YaTC/YaTC_pretrained_model.pth")['model']
    # yes without the 's'
    yatc_frozen_model = models_YaTC.__dict__['TraFormer_YaTC'](
            num_classes=n_classes,
            drop_path_rate=0.1,
        )
    interpolate_pos_embed(yatc_frozen_model, checkpoint_model)

    #rename norm to fc_norm and delete extra keys
    checkpoint_model['fc_norm.bias'] = checkpoint_model['norm.bias']
    checkpoint_model['fc_norm.weight'] = checkpoint_model['norm.weight']

    keys_to_del = ['mask_token', 'norm.weight', 'norm.bias']
    for key in checkpoint_model.keys():
        if key.startswith('decoder'):
            keys_to_del.append(key)

    for key in keys_to_del:
        del checkpoint_model[key]

    yatc_frozen_model.load_state_dict(checkpoint_model, strict=False)

    yatc_models = {}
    for i in range(gpus):
        yatc_models[i] = copy.deepcopy(yatc_frozen_model)
        yatc_models[i].to(f"cuda:{i}")

    print(f"Total: {len(loader_yatc)}")

    def encode_and_append(batch, model, result_list, i):
        imgs, _ = batch
        imgs = imgs.to(f"cuda:{i}")
        with torch.no_grad():
            result_list.append(model.forward_features(imgs).mean(dim=1).cpu())
        del imgs
        del batch

    counter = 0
    result_embeddings = []
    result_filenames = []
    with torch.no_grad():
        iterator = iter(loader_yatc)
        try:
            for y in tqdm(range(0, min(len(loader_yatc) // gpus, limit))):
                yatc_emb = defaultdict(list)
                batches = [next(iterator) for i in range(gpus)]
                threads = []
                for i in range(gpus):
                    t = threading.Thread(target=encode_and_append, args=(batches[i], yatc_models[i], yatc_emb[i], i))
                    t.start()
                    threads.append(t)
                for t in threads:
                    t.join()
                del batches
                result_embeddings.append(torch.cat([torch.cat(yatc_emb[i]) for i in yatc_emb]))
                result_filenames.extend([x[0] for x in loader_yatc.dataset.samples[y*gpus*batch_size:(y+1)*gpus*batch_size]])
                
        except StopIteration:
            print("finished")
        except Exception as e:
            print(e)

    return torch.cat(result_embeddings), result_filenames


# In[ ]:


caida_emb = get_embeddings("../data/newYaTC/", n_classes=4, batch_size=1, gpus=1)
with open("../data/newYaTC/synth_emb.pkl", "bw") as f:
    pickle.dump(caida_emb, f)


# In[ ]:


caida_emb = get_embeddings("../data/caida/", n_classes=1, batch_size=512)
with open("../data/caida_emb.pkl", "bw") as f:
    pickle.dump(caida_emb, f)


# In[ ]:


caida_emb = get_embeddings("../data/cicapt/", n_classes=22, batch_size=512)
with open("../data/cicapt_emb.pkl", "bw") as f:
    pickle.dump(caida_emb, f)


# In[ ]:


cicids_emb = get_embeddings("../data/cicids/", n_classes=8, batch_size=512)
with open("../data/cicids_emb.pkl", "bw") as f:
    pickle.dump(cicids_emb, f)


# In[ ]:


cross_emb = get_embeddings("../data/cross/", n_classes=210, batch_size=512)
with open("../data/cross_emb.pkl", "bw") as f:
    pickle.dump(cross_emb, f)


# In[ ]:


mawi_emb = get_embeddings("../data/mawi/", n_classes=1, batch_size=512)
with open("../data/mawi_emb.pkl", "bw") as f:
    pickle.dump(mawi_emb, f)


# In[ ]:


emb = get_embeddings("../data/yatc/", n_classes=1, batch_size=1, gpus=1)
with open("../data/synthstability_emb.pkl", "bw") as f:
    pickle.dump(emb, f)


# In[ ]:


emb = get_embeddings("../data/synth/", n_classes=5, batch_size=1, gpus=1)
with open("../data/synth_emb.pkl", "bw") as f:
    pickle.dump(emb, f)


# In[ ]:


emb = get_embeddings("../data/perf/", n_classes=4, batch_size=1, gpus=1)
with open("../data/perf_emb.pkl", "bw") as f:
    pickle.dump(emb, f)


# In[ ]:




