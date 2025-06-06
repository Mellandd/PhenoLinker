import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
from torch_geometric.data import HeteroData
import numpy as np

# Some files are not uploaded to the repository for its size. Read README to download them in this folder.
genes = 'genes.csv'
phenotypes = 'phenotypes.csv'
gene_edges = 'phenotypes_to_genes.tsv'
phen_edges = 'phenotype_edges.csv'

# We filter the genes in phenotype_to_genes.tsv (from HPO), to only use
# genes with features from genes.csv

df_genes = pd.read_csv(genes)
df_gene_edges = pd.read_csv(gene_edges, sep='\t')
df_gene_edges = df_gene_edges[df_gene_edges['entrez-gene-symbol'].isin(df_genes['gene'].to_list())]
df_gene_edges.to_csv(gene_edges, sep='\t', index=False)

class SequenceEncoder(object):
    def __init__(self, model_name='pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb', device=None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df):
        x = self.model.encode(df.values, show_progress_bar=True,
                              convert_to_tensor=True, device=self.device)
        return x.cpu()

class IdentityEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)    

class ConstantEncoder(object):
    def __init__(self, dtype=None):
        self.dtype = dtype
        
    def __call__(self, df):
        ct = np.asarray([1.0 for _ in range(len(df.values))])
        return torch.from_numpy(ct).view(-1, 1).to(self.dtype)

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


def load_edge_csv(path, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    df = pd.read_csv(path, **kwargs)

    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])

    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr

pheno_x, pheno_mapping = load_node_csv(
    phenotypes, index_col='Phenotypes', encoders={
        'Definition': SequenceEncoder()
    })

gene_x, gene_mapping = load_node_csv(
    genes, index_col='gene', encoders={
          "ExACpLI":IdentityEncoder(),
          "ExACpRec":IdentityEncoder(),
          "ExACpNull":IdentityEncoder(),
          "ExACpMiss":IdentityEncoder(),
          "ExprSpecificAdiposeSub":IdentityEncoder(),
          "ExprSpecificAdiposeVisceral":IdentityEncoder(),
          "ExprSpecificAdrenalGland":IdentityEncoder(),
          "ExprSpecificArteryAorta":IdentityEncoder(),
          "ExprSpecificArteryCoronary":IdentityEncoder(),
          "ExprSpecificArteryTibial":IdentityEncoder(),
          "ExprSpecificAmygdala":IdentityEncoder(),
          "ExprSpecificAntCingCortex":IdentityEncoder(),
          "ExprSpecificCaudate":IdentityEncoder(),
          "ExprSpecificCerebHemisphere":IdentityEncoder(),
          "ExprSpecificCerebellum":IdentityEncoder(),
          "ExprSpecificCortex":IdentityEncoder(),
          "ExprSpecificFCortex":IdentityEncoder(),
          "ExprSpecificHippocampus":IdentityEncoder(),
          "ExprSpecificHypothalamus":IdentityEncoder(),
          "ExprSpecificNucAccumbens":IdentityEncoder(),
          "ExprSpecificPutamen":IdentityEncoder(),
          "ExprSpecificSpinalcord":IdentityEncoder(),
          "ExprSpecificSubstantianigra":IdentityEncoder(),
          "ExprSpecificBreast":IdentityEncoder(),
          "ExprSpecificCellsLymphocytes":IdentityEncoder(),
          "ExprSpecificCellsFirbroblasts":IdentityEncoder(),
          "ExprSpecificColonSigmoid":IdentityEncoder(),
          "ExprSpecificColonTransverse":IdentityEncoder(),
          "ExprSpecificEsophGastJunction":IdentityEncoder(),
          "ExprSpecificEsophMucosa":IdentityEncoder(),
          "ExprSpecificEsophMuscularis":IdentityEncoder(),
          "ExprSpecificHeartAtrialApp":IdentityEncoder(),
          "ExprSpecificHeartLeftVent":IdentityEncoder(),
          "ExprSpecificLiver":IdentityEncoder(),
          "ExprSpecificLung":IdentityEncoder(),
          "ExprSpecificMuscleSkeletal":IdentityEncoder(),
          "ExprSpecificNerveTibial":IdentityEncoder(),
          "ExprSpecificOvary":IdentityEncoder(),
          "ExprSpecificPancreas":IdentityEncoder(),
          "ExprSpecificPituitary":IdentityEncoder(),
          "ExprSpecificProstate":IdentityEncoder(),
          "ExprSpecificSkinSuprapubic":IdentityEncoder(),
          "ExprSpecificSkinLowerLeg":IdentityEncoder(),
          "ExprSpecificSmallIntestine":IdentityEncoder(),
          "ExprSpecificSpleen":IdentityEncoder(),
          "ExprSpecificStomach":IdentityEncoder(),
          "ExprSpecificTestis":IdentityEncoder(),
          "ExprSpecificThyroid":IdentityEncoder(),
          "ExprSpecificUterus":IdentityEncoder(),
          "ExprSpecificVagina":IdentityEncoder(),
          "ExprSpecificWholeBlood":IdentityEncoder(),
          "CountsOverlap":IdentityEncoder(),
          "CountsProtCodOverlap":IdentityEncoder(),
          "StringCombined":IdentityEncoder(),
}, sep=',')

edge_index, edge_label = load_edge_csv(
    gene_edges,
    src_index_col='HPO-id', 
    src_mapping=pheno_mapping,
    dst_index_col='entrez-gene-symbol',
    dst_mapping=gene_mapping,
    sep='\t'
)

data = HeteroData()

data['phenotype'].x = pheno_x
data['gene'].x = gene_x

data['phenotype','related_to','gene'].edge_index = edge_index
data['phenotype','related_to','gene'].edge_label = edge_label

edge_index, edge_label = load_edge_csv(
    phen_edges,
    src_index_col='Phenotype', 
    src_mapping=pheno_mapping,
    dst_index_col='is_a',
    dst_mapping=pheno_mapping,
)

data['phenotype','is_a','phenotype'].edge_index = edge_index
data['phenotype','is_a','phenotype'].edge_label = edge_label

torch.save(data, 'dataframe.pt')
