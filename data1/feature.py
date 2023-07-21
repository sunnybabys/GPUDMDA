import itertools
import pandas as pd
import numpy as np
import gate_feature
from data_preprocess import single_generate_graph_adj_and_feature
from sim_processing import sim_thresholding


association = pd.read_csv("./trans_data/data2/index_association.csv", header=0, index_col=0).values

#non-linear feature
m_threshold = [0.5]  # generate subgraph
d_threshold = [0.5]
epochs=[400]
m_sim = pd.read_csv("./trans_data/data2/microbe_features.txt", header=None, index_col=None ,sep='\t').values
d_sim = pd.read_csv("./trans_data/data2/disease_features.txt", header=None, index_col=None ,sep='\t').values



for s in itertools.product(m_threshold,d_threshold,epochs):
    # GATE
    print(s[0],s[1])
    m_network = sim_thresholding(m_sim, s[0])
    d_network = sim_thresholding(d_sim, s[1])
    m_adj, m_features = single_generate_graph_adj_and_feature(m_network, association.T)
    d_adj, d_features = single_generate_graph_adj_and_feature(d_network, association)
    m_embeddings = gate_feature.get_gate_feature(m_adj, m_features,s[2], 1)
    d_embeddings = gate_feature.get_gate_feature(d_adj, d_features,s[2], 1)

m_emb = pd.DataFrame(m_embeddings)
m_emb.to_csv('./trans_data/data2/nonm_feature.csv',header=None,index=None)
d_emb = pd.DataFrame(d_embeddings)
d_emb.to_csv('./trans_data/data2/nond_feature.csv',header=None,index=None)




print("Finished")