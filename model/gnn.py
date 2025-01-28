import torch
from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

# Final classifier layer. Generates the score for a gene-phenotype edge.
class Classifier(torch.nn.Module):
    def forward(self, x_pheno: Tensor, x_gene: Tensor, edge_label_index: Tensor) -> Tensor:
        edge_feat_pheno = x_pheno[edge_label_index[0]]
        edge_feat_gene = x_gene[edge_label_index[1]]
        return torch.sigmoid((edge_feat_pheno * edge_feat_gene).sum(dim=-1))
        
# Heterogeneous GNN. Uses HeteroConv layers with SAGEConv.
class HeteroGNN(torch.nn.Module): 
    def __init__(self, hidden_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('phenotype', 'is_a', 'phenotype'): SAGEConv((-1, -1), hidden_channels, aggr='mean'),
                ('phenotype', 'related_to', 'gene'): SAGEConv((-1, -1), hidden_channels, aggr='mean'),
                ('gene', 'rev_related_to', 'phenotype'): SAGEConv((-1, -1), hidden_channels, aggr='mean'),
            }, aggr='sum')
            self.convs.append(conv)
        self.final = HeteroConv({
                ('phenotype', 'is_a', 'phenotype'): SAGEConv((-1, -1), hidden_channels, aggr = 'mean'),
                ('phenotype', 'related_to', 'gene'): SAGEConv((-1, -1), hidden_channels, aggr = 'mean'),
                ('gene', 'rev_related_to', 'phenotype'): SAGEConv((-1, -1), hidden_channels, aggr = 'mean'), # add_self_loops=False,
            }, aggr='sum')
        self.classifier = Classifier()
        
    def encode(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        x_dict = self.final(x_dict, edge_index_dict)
        return x_dict
    
    def decode(self, x_dict, edge_label_index):
        pred = self.classifier(
            x_dict["phenotype"],
            x_dict["gene"],
            edge_label_index,
        )
        return pred

    def forward(self, x_dict, edge_index_dict, edge_label_index):
        x_dict = self.encode(x_dict, edge_index_dict)
        pred = self.decode(x_dict, edge_label_index)
        return pred
