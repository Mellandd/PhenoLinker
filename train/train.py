import tqdm
import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, PrecisionRecallDisplay
import torch_geometric.transforms as T
from torch_geometric.loader import LinkNeighborLoader
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from matplotlib.ticker import MaxNLocator
from torch_geometric import seed_everything

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')))
from gnn import *

# Load graph data
seed_everything(1)
data = torch.load('../data/dataframe.pt')
data = T.ToUndirected()(data)
data["phenotype"].x = data["phenotype"].x.to(torch.float32)
data["gene"].x = data["gene"].x.to(torch.float32)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Generate train, eval, test splits
transform = T.RandomLinkSplit(
    num_val=0,
    num_test=0,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=4.0,
    edge_types=("phenotype", "related_to", "gene"),
    rev_edge_types=("gene", "rev_related_to", "phenotype"), 
)
train_data, val_data, test_data = transform(data)

# Load the model
model = HeteroGNN(hidden_channels=64, num_layers=3)
model = model.to(device)

optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

# Generate the train and eval loaders to use during the training
edge_label_index = train_data["phenotype", "related_to", "gene"].edge_label_index
edge_label = train_data["phenotype", "related_to", "gene"].edge_label
train_loader = LinkNeighborLoader(
    data=train_data,
    num_neighbors=[-1, -1],
    edge_label_index=(("phenotype", "related_to", "gene"), edge_label_index),
    edge_label=edge_label,
    batch_size=2048,
    shuffle=True,
)

edge_label_index = val_data["phenotype", "related_to", "gene"].edge_label_index
edge_label = val_data["phenotype", "related_to", "gene"].edge_label
val_loader = LinkNeighborLoader(
    data=val_data,
    num_neighbors=[-1, -1],
    edge_label_index=(("phenotype", "related_to", "gene"), edge_label_index),
    edge_label=edge_label,
    batch_size=2048,
    shuffle=False,
)


# Train for 15 epochs
losses = []
auprs = []
auprs_eval = []
for epoch in range(1,16):
    preds = []
    ground_truths = []
    total_loss = total_examples = 0
    for sampled_data in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        sampled_data.to(device)
        pred = model(sampled_data.x_dict, sampled_data.edge_index_dict, sampled_data["phenotype", "related_to", "gene"].edge_label_index)
        ground_truth = sampled_data["phenotype", "related_to", "gene"].edge_label
        loss = F.binary_cross_entropy(pred, ground_truth)
        loss.backward()
        optimizer.step()
        preds.append(pred.detach())
        ground_truths.append(ground_truth)
        total_loss += float(loss) * pred.numel()
        total_examples += pred.numel()
    pred = torch.cat(preds, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
    aupr = average_precision_score(ground_truth, pred)
    auprs.append(aupr)
    print(f"Epoch: {epoch:03d}, Loss: {total_loss / total_examples:.4f}")
    print(f"AUCPR: {aupr}")
    losses.append(total_loss/ total_examples)
    preds_eval = []
    ground_truths_eval = []
    for sampled_data in tqdm.tqdm(val_loader):
        with torch.no_grad():
            sampled_data.to(device)
            ground_truth = sampled_data["phenotype", "related_to", "gene"].edge_label
            pred = model(sampled_data.x_dict, sampled_data.edge_index_dict, sampled_data["phenotype", "related_to", "gene"].edge_label_index)
            preds_eval.append(pred)
            ground_truths_eval.append(ground_truth)
    pred = torch.cat(preds_eval, dim=0).cpu().numpy()
    ground_truth = torch.cat(ground_truths_eval, dim=0).cpu().numpy()
    auc = roc_auc_score(ground_truth, pred)
    aupr = average_precision_score(ground_truth, pred)
    auprs_eval.append(aupr)
    print(f"aupr eval: {aupr}")
        
torch.save(model, '../data/model.pt')

# Generate the test loader
edge_label_index = test_data["phenotype", "related_to", "gene"].edge_label_index
edge_label = test_data["phenotype", "related_to", "gene"].edge_label

test_loader = LinkNeighborLoader(
    data=test_data,
    num_neighbors=[-1, -1],
    edge_label_index=(("phenotype", "related_to", "gene"), edge_label_index),
    edge_label=edge_label,
    batch_size=2048,
    shuffle=False,
)

# After training evaluate in test data
preds = []
ground_truths = []
for sampled_data in tqdm.tqdm(test_loader):
    with torch.no_grad():
        sampled_data.to(device)
        pred = model(sampled_data.x_dict, sampled_data.edge_index_dict, sampled_data["phenotype", "related_to", "gene"].edge_label_index)
        ground_truth = sampled_data["phenotype", "related_to", "gene"].edge_label
        preds.append(pred)
        ground_truths.append(ground_truth)
pred = torch.cat(preds, dim=0).cpu().numpy()
ground_truth = torch.cat(ground_truths, dim=0).cpu().numpy()
pred_classes = np.ndarray.round(pred)

# Test metrics

print(f"F1: {f1_score(ground_truth, pred_classes)}")
print(f"F1 micro: {f1_score(ground_truth, pred_classes, average='micro')}")
print(f"F1 macro: {f1_score(ground_truth, pred_classes, average='macro')}")
print(f"Precision: {precision_score(ground_truth, pred_classes)}")
print(f"Recall: {recall_score(ground_truth, pred_classes)}")
print(f"ROC AUC: {roc_auc_score(ground_truth, pred)}")
print(f"PR AUC {average_precision_score(ground_truth, pred)}")
print(f"PR AUC micro {average_precision_score(ground_truth, pred, average='micro')}")
print(f"PR AUC macro {average_precision_score(ground_truth, pred, average='macro')}")
print(classification_report(ground_truth, pred_classes))
cm = confusion_matrix(ground_truth, pred_classes)
print(cm)
