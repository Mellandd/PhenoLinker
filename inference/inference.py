import torch
from torch import Tensor
import torch_geometric.transforms as T
import pandas as pd
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

with torch.no_grad():  
   x_dict = model.encode(data.x_dict, data.edge_index_dict)

edge_label_index = data["phenotype", "related_to", "gene"].edge_index

# Explainability example: Gene SNCA with phenotype HP:0100315

index = torch.tensor([[mapping_gene['SNCA']], [mapping_phen['HP:0100315']]])
edge_label_index = (("phenotype", "related_to", "gene"), index)

score = model.decode(x_dict, edge_label_index)

print(score)

