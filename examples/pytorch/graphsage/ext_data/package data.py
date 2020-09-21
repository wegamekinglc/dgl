#!/usr/bin/env python
# coding: utf-8

# In[1]:


import json
from scipy.sparse.coo import coo_matrix
import scipy.sparse as sp
import numpy as np


# In[34]:


with open('ppi-G.json') as fp:
    graph = json.load(fp)
    
with open('ppi-feats.npy', 'rb') as fp:
    features = np.load(fp)
    
with open('ppi-class_map.json') as fp:
    class_map = json.load(fp)
    


# In[32]:


nodes = graph['nodes']
edges = graph['links']
num_nodes = len(nodes)
node_types = np.zeros(num_nodes, dtype=int)
label = np.zeros(num_nodes, dtype=int)
feature = features


# In[38]:


for i, node in enumerate(nodes):
    if node['test'] == True:
        node_types[i] = 3
    elif node['val'] == True:
        node_types[i] = 2
    else:
        node_types[i] = 1
    label[i] = class_map[str(i)][0] 

row_index = []
col_index = []
data_values = []
previous_source = -1

metioned_nodes = set()
for edge in edges:
    if edge['source'] != previous_source:
        previous_source = edge['source']
        row_index.append(previous_source)
        col_index.append(previous_source)
        data_values.append(1.0)
        metioned_nodes.add(previous_source)
    
    if edge['source'] < edge['target']:
        row_index.append(edge['source'])
        col_index.append(edge['target'])
        data_values.append(1.0)
        
        row_index.append(edge['target'])
        col_index.append(edge['source'])
        data_values.append(1.0)
        
for index in range(num_nodes):
    if index not in metioned_nodes:
        row_index.append(index)
        col_index.append(index)
        data_values.append(1.0)


# In[39]:


coo_adj = coo_matrix((data_values, (row_index, col_index)), shape=(num_nodes, num_nodes))


# In[40]:


with open('graph.npz', 'wb') as fp:
    sp.save_npz(fp, coo_adj)


# In[41]:


with open('data.npz', 'wb') as fp:
    np.savez_compressed(fp, **{'feature': feature, 'label': label, 'node_types': node_types}, allow_pickle=True)


# In[42]:


coo_adj


# In[ ]:




