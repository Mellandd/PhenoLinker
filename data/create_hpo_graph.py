import networkx as nx
import obonet
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

url = 'https://github.com/obophenotype/human-phenotype-ontology/releases/download/v2022-12-15/hp.obo'
graph = obonet.read_obo(url)

descr = []
for node in graph.nodes():
    if 'def' in graph.nodes[node]:
        descr.append(graph.nodes[node]['def'])
    else:
        descr.append(graph.nodes[node]['name'])

for node in graph.nodes():
    node_attrs = dict(graph.nodes[node])  # create a copy of the attributes
    for attr in node_attrs.keys():
        del graph.nodes[node][attr]
        
phen = [x[0] for x in graph.edges]
is_a = [x[1] for x in graph.edges]

df = pd.DataFrame(graph.nodes(), columns=['Phenotypes'])
df['Definition'] = descr
df.to_csv('phenotypes.csv', index=False)

df2 = pd.DataFrame({'Phenotype': phen, 'is_a': is_a})
df2.to_csv('phenotype_edges.csv', index=False)