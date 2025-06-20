import argparse

import json
from loader import MoleculeDataset_aug
from torch_geometric.data import DataLoader
from torch_geometric.nn.inits import uniform
from torch_geometric.nn import global_mean_pool
from loader import MoleculeDataset

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
from tqdm.contrib import tzip
import numpy as np

from model import GNN
from sklearn.metrics import roc_auc_score

import time
from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

# from tensorboardX import SummaryWriter

from copy import deepcopy


def cycle_index(num, shift):
    arr = torch.arange(num) + shift
    arr[-shift:] = torch.arange(shift)
    return arr

class Discriminator(nn.Module):
    def __init__(self, hidden_dim):
        super(Discriminator, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        size = self.weight.size(0)
        uniform(size, self.weight)

    def forward(self, x, summary):
        h = torch.matmul(summary, self.weight)
        return torch.sum(x*h, dim = 1)


class graphcl(nn.Module):

    def __init__(self, gnn):
        super(graphcl, self).__init__()
        self.gnn = gnn
        self.pool = global_mean_pool
        self.projection_head = nn.Sequential(nn.Linear(300, 300), nn.ReLU(inplace=True), nn.Linear(300, 300))

    def forward_cl(self, x, edge_index, edge_attr, batch):
        x = self.gnn(x, edge_index, edge_attr)
        x = self.pool(x, batch)
        x = self.projection_head(x)
        return x

    def loss_cl(self, x1, x2):
        T = 0.1
        batch_size, _ = x1.size()
        x1_abs = x1.norm(dim=1)
        x2_abs = x2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()
        return loss


def train(args, model, device, dataset, optimizer, epoch):

    dataset.aug = "none"
    dataset1 = dataset.shuffle()
    dataset2 = deepcopy(dataset1)
    dataset1.aug, dataset1.aug_ratio = args.aug1, args.aug_ratio1
    dataset2.aug, dataset2.aug_ratio = args.aug2, args.aug_ratio2

    loader1 = DataLoader(dataset1, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)
    loader2 = DataLoader(dataset2, batch_size=args.batch_size, num_workers = args.num_workers, shuffle=False)

    model.train()

    train_acc_accum = 0
    train_loss_accum = 0
    epoch_iterator = tqdm(tzip(loader1, loader2), desc=f"Training (Epoch {epoch}/{args.epochs})")
    for step, batch in enumerate(epoch_iterator):
        batch1, batch2 = batch
        batch1 = batch1.to(device)
        batch2 = batch2.to(device)

        optimizer.zero_grad()

        x1 = model.forward_cl(batch1.x, batch1.edge_index, batch1.edge_attr, batch1.batch)
        x2 = model.forward_cl(batch2.x, batch2.edge_index, batch2.edge_attr, batch2.batch)
        loss = model.loss_cl(x1, x2)

        loss.backward()
        optimizer.step()

        train_loss_accum += float(loss.detach().cpu().item())
        #acc = (torch.sum(positive_score > 0) + torch.sum(negative_score < 0)).to(torch.float32)/float(2*len(positive_score))
        acc = torch.tensor(0)

        #threshold=0.7
        #similarity = torch.nn.functional.cosine_similarity(x1, x2, dim=-1)
        #correct = (similarity > threshold).float()
        #acc = correct.sum() / len(correct)

        train_acc_accum += float(acc.detach().cpu().item())

    return train_acc_accum/(step+1), train_loss_accum/(step+1)


def eval(args, model, device, loader, only_pred=False):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)

        y_one_hot = torch.nn.functional.one_hot(batch.y, num_classes=2).to(torch.float32)
        #y_true.append(batch.y.view(pred.shape))
        y_true.append(y_one_hot)
        y_scores.append(pred)
    
    if only_pred:
        return y_scores
    
    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            #is_valid = y_true[:,i]**2 > 0
            #roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
            is_valid = (y_true[:, i] == 0) | (y_true[:, i] == 1)
            roc_list.append(roc_auc_score(y_true[is_valid, i], y_scores[is_valid, i]))
    
    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--dataset', type=str, default = 'cora', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--output_model_file', type = str, default = '', help='filename to output the pre-trained model')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--seed', type=int, default=0, help = "Seed for splitting dataset.")
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--aug1', type=str, default = 'none')
    parser.add_argument('--aug_ratio1', type=float, default = 0.2)
    parser.add_argument('--aug2', type=str, default = 'none')
    parser.add_argument('--aug_ratio2', type=float, default = 0.2)
    args = parser.parse_args()


    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    #set up dataset
    import os
    PATH = f'''{os.getcwd()}/datasets/'''.replace('easy', 'data_processing')
    dataset = MoleculeDataset_aug(PATH + args.dataset, dataset=args.dataset)
    #dataset = torch.load(f'''dataset/{source}/{source}_data_full.pt''', weights_only=False)
    print("data",dataset)
    print(dataset[0])

    #set up model
    gnn = GNN(args.num_layer, args.emb_dim, JK = args.JK, drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type)

    model = graphcl(gnn)
    
    model.to(device)

    #set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    print("optim",optimizer)

    start = time.time()
    for epoch in range(1, args.epochs):
        print("====epoch " + str(epoch))
    
        train_acc, train_loss = train(args, model, device, dataset, optimizer, epoch)

        print('train_acc',train_acc)
        print('train_loss',train_loss)

        if epoch % 20 == 0:
            PATH = f'''{os.getcwd()}/'''
            torch.save(gnn.state_dict(), PATH + "models_graphcl/graphcl_" + args.dataset + "_" + str(epoch) + ".pth")
    
    end = time.time()
    print(f"Time to train {args.dataset}: {end-start}")
    PATH = f'''{os.getcwd()}/'''
    torch.save(gnn.state_dict(), PATH + "models_graphcl/graphcl_" + args.dataset + "_final.pth")

    # Get predictions for source test dataset
    #PATH = f'''{os.getcwd()}/datasets/'''.replace('easy', 'data_processing')
    #test_dataset = MoleculeDataset(PATH + args.dataset, dataset=args.dataset, fold=3)
    #test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    #logits = eval(args, model, device, test_loader, only_pred=True)
    #train_acc = eval(args, model, device, test_loader)
    #predictions = [torch.sigmoid(tensor).cpu().numpy().tolist() for tensor in logits]  # Bring me some probabilities!

    # Save to a JSON file
    #path = os.getcwd()
    #try:
    #    os.mkdir(path + "/probabilities")
    #except:
    #    print("probabilities folder already created")

    #with open(path + f"/probabilities/{args.dataset}_probabilities.json", "w") as json_file:
    #    json.dump(predictions, json_file)

    with open(f'outputs/{args.dataset}_pretrain_result.log', 'a+') as f:
        f.write('train_acc: ' + str(train_acc))
        f.write('time: ' + str(end-start))
        f.write('\n')
    
if __name__ == "__main__":
    main()
