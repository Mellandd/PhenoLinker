from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import HeteroData
import numpy as np
import pandas as pd
import pickle

phenotypes = '/home/jlmellina/tfm/src/data/phenotypes.csv'

class SequenceEncoder(object):
    def __init__(self, model_name='pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()
    
def load_node_csv(path, index_col, encoders=None, **kwargs):
    df = pd.read_csv(path, index_col=index_col, **kwargs)
    mapping = {index: i for i, index in enumerate(df.index.unique())}

    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)
    else: 
        xs = [np.ones(len(df['gene']))]
        x = torch.cat(xs, dim=-1)

    return x, mapping

pheno_x, pheno_mapping = load_node_csv(
    phenotypes, index_col='Phenotypes', encoders={
        'Definition': SequenceEncoder()
    })

df = pd.DataFrame(pheno_x.numpy())

df.to_csv('phen.csv', index=False)

with open('phen_map.pck', 'wb') as file:
    pickle.dump(pheno_mapping, file, protocol=pickle.HIGHEST_PROTOCOL)