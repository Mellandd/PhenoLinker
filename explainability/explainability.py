from torch_geometric.explain import Explainer, CaptumExplainer
import torch
import torch_geometric.transforms as T
import captum
import pandas as pd
import matplotlib.pyplot as plt

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
from gnn import *

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = torch.load('../data/dataframe.pt')
data = T.ToUndirected()(data)
data["phenotype"].x = data["phenotype"].x.to(torch.float32)
data["gene"].x = data["gene"].x.to(torch.float32)
data = data.to(device)
    
genes = '../data/genes.csv'
phenotypes = '../data/phenotypes.csv'
gen_edges = '../data/phenotypes_to_genes.csv'
phen_edges = '../data/phenotype_edges.csv'

df_gen = pd.read_csv(genes, index_col='gene')
mapping_gene = {index: i for i, index in enumerate(df_gen.index.unique())}
mapping_gene_reverse = {i: index for i, index in enumerate(df_gen.index.unique())}
df_phen = pd.read_csv(phenotypes, index_col='Phenotypes')
mapping_phen = {index: i for i, index in enumerate(df_phen.index.unique())}
mapping_phen_reverse = {i: index for i, index in enumerate(df_phen.index.unique())}

model = torch.load('../data/model.pt')
model = model.to(device)

# Example of interpretability: Gene SNCA and phenotype HP:0100315

index = torch.tensor([[mapping_gene['SNCA']], [mapping_phen['HP:0100315']]])
edge_label_index = (("phenotype", "related_to", "gene"), index)

explainer = Explainer(
    model=model,
    algorithm=CaptumExplainer('IntegratedGradients'),
    explanation_type='model',
    model_config=dict(
        mode='regression',
        task_level='edge',
        return_type='raw',
    ),
    node_mask_type='attributes',
    edge_mask_type='object',
    threshold_config=dict(
        threshold_type='topk',
        value=20,
    ),
)
explanation = explainer(
    data.x_dict,
    data.edge_index_dict,
    index = index,
    edge_label_index = edge_label_index
)
print(f'Generated model explanations in {explanation.available_explanations}')

# path to store the bar plot
path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)