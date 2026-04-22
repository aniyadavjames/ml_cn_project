#!/usr/bin/env python
# coding: utf-8

# ## Stability measurement

# In[1]:


data = {
    "etbert": "../data/etbert/synth_emb.pkl",
    "yatc": "../data/yatc/synth_emb.pkl",
    "netmamba": "../data/netmamba/netmamba_synth_emb.pkl",
    "netfound": "../data/netfound/synth_emb.pkl",
}

mapping = lambda x: x.split("/")[-1].split("exp")[0]


# In[2]:


import torch
import torch.nn.functional as F
import pickle
import os
import pandas as pd

def measure_spread(tensor1, tensor2):
    similarity_matrix = F.cosine_similarity(tensor1.unsqueeze(1), tensor2.unsqueeze(0), dim=2)
    return similarity_matrix.mean()

def measure_dist(tensor1: torch.Tensor, tensor2: torch.Tensor) -> float:
    diff = torch.abs(tensor1.unsqueeze(1) - tensor2.unsqueeze(0))
    distances = diff.sum(dim=2)
    return distances.mean().item()


# In[3]:


def linear_probing(embeddings, labels, test_size=0.2, random_state=42):
    import torch
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    # Convert embeddings tensor to numpy array
    X = embeddings.cpu().numpy()
    y = np.array(labels)
    
    # Ensure exactly two classes exist
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        raise ValueError("There must be exactly two classes in the labels.")
    
    # Map string labels to binary integers
    class_to_int = {cls: idx for idx, cls in enumerate(unique_classes)}
    y_int = np.array([class_to_int[label] for label in y])
    
    # Shuffle and split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=test_size, random_state=random_state, shuffle=True
    )
    
    # Train logistic regression classifier
    clf = LogisticRegression(random_state=random_state, max_iter=1000)
    clf.fit(X_train, y_train)
    
    # Evaluate classifier performance
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    return train_f1, test_f1


# In[4]:


"""
experiments:
'fifo_6m_bbr_prof50_36_' - baseline
'fifo_6m_cubic_prof50_36_' - different cc algorithm
'codel_6m_bbr_prof50_36_' - different AQM
'fifo_6m_bbr_prof72_29_' - different cross traffic
'codel_6m_cubic_prof72_29_' - different cc, aqm, and cross traffic
"""

baseline = 'fifo_6m_bbr_prof50_36_'
others = {
    'fifo_6m_cubic_prof50_36_': "Congestion Control",
    'codel_6m_bbr_prof50_36_': "AQM",
    'fifo_6m_bbr_prof72_29_': "Crosstraffic",
    'codel_6m_cubic_prof72_29_': "All",
}

results = {
    'yatc': {
        'Average': 0.8633,
    },
    'etbert': {
        'Average': 0.7977,
    },
    'netmamba': {
        'Average': 0.9639,
    },
    'netfound': {
        'Average': 0.8017,
    },
}

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

    # self-similarity for baseline - stability test
    results[model]['Stability baseline'] = measure_spread(classes[baseline], classes[baseline]).item()
    print(f"Model {model}, baseline stability: avg_cosine_similarity = {results[model]['Stability baseline']:.4f}, avg_L1_dist = {measure_dist(classes[baseline], classes[baseline]):.0f}")

    # similarity of others
    for x in others:
        avg_cos_sim = measure_spread(classes[baseline], classes[x])
        results[model][others[x]] = avg_cos_sim.item()
        avg_dist = measure_dist(classes[baseline], classes[x])
        f1_train, f1_test = linear_probing(
            embeddings=torch.cat([classes[baseline], classes[x]]),
            labels=[baseline] * classes[baseline].size(0) + [x] * classes[x].size(0),
        )
        results[model][others[x] + "_f1"] = f1_test
        print(f"Model {model}, class {x}: avg_cosine_similarity = {avg_cos_sim:.4f}, avg_L1_dist = {avg_dist:.0f}, Train F1 Score: {f1_train:.4f}, Test F1 Score: {f1_test:.4f}")
        
        


# In[5]:


results


# In[6]:


model_order = ['yatc', 'etbert', 'netfound', 'netmamba']
row_order = ['Congestion Control', 'AQM', 'Crosstraffic', 'All']
cos_results = {x: {
    y: results[x][y] for y in row_order
} for x in model_order}
f1_results = {x + "_f1": {
    y: results[x].get(y + "_f1", '-') for y in row_order
} for x in model_order}
df_last3_copy = pd.DataFrame(cos_results | f1_results)[['yatc', 'yatc_f1', 'etbert', 'etbert_f1', 'netfound', 'netfound_f1', 'netmamba', 'netmamba_f1']]
df_last3_copy

average = {x: results[x]['Average'] for x in model_order}
stability = {x: results[x]['Stability baseline'] for x in model_order}

# --- DataFrame 1: Delta from Stability (sign reversed: value - stability) ---
# Copy all columns from the original last three rows, update only odd columns.
df_delta_all = df_last3_copy.copy()
for col in model_order:
    df_delta_all[col] = df_last3_copy[col] - stability[col]

# --- DataFrame 2: Normalized Delta in Percentage (sign reversed) ---
# Again, copy all columns, and update only the odd ones.
df_norm_all = df_last3_copy.copy()
for col in model_order:
    # The denominator is (stability - average)
    range_val = stability[col] - average[col]
    df_norm_all[col] = ((df_last3_copy[col] - stability[col]) / range_val) * 100

# --- Helper function to convert a DataFrame to LaTeX table rows ---
def df_to_latex_rows(df, percent_cols=None, float_format="{:.4f}"):
    """
    Converts each row of the DataFrame into a LaTeX table row string.
    :param df: DataFrame whose rows will be converted.
    :param percent_cols: List of columns that should be formatted as percentages.
    :param float_format: Format string for numeric values.
    """
    if percent_cols is None:
        percent_cols = []
    latex_lines = []
    for i, row in df.iterrows():
        row_values = []
        for col in df.columns:
            if col == 'Metric':
                row_values.append(str(row[col]))
            else:
                if col in percent_cols:
                    try:
                        # Format as percentage with 2 decimal places and add a percent sign.
                        formatted_val = f"{row[col]:.2f}\\%"
                        formatted_val = f"{row[col]:.2f}\\\\%"
                    except Exception as e:
                        formatted_val = str(row[col])
                else:
                    try:
                        formatted_val = float_format.format(row[col])
                    except Exception as e:
                        formatted_val = str(row[col])
                row_values.append(formatted_val)
        # Build the LaTeX row with a '%' at the beginning (like your input)
        latex_line = "%\t" + " & ".join(row_values) + " \\\\"
        latex_lines.append(latex_line)
    return "\n".join(latex_lines)

# --- Generate LaTeX rows ---
# For DataFrame 1, odd columns are standard numeric values.
latex_df_delta = df_to_latex_rows(df_delta_all, percent_cols=[], float_format="{:.4f}")

# For DataFrame 2, format odd columns as percentages.
latex_df_norm = df_to_latex_rows(df_norm_all, percent_cols=model_order, float_format="{:.2f}")

# --- Print the LaTeX rows ---
print("LaTeX rows for DataFrame 1 (Delta from Stability):")
print(latex_df_delta)
print("\nLaTeX rows for DataFrame 2 (Normalized Delta in %):")
print(latex_df_norm)


# In[7]:


average


# In[8]:


stability

