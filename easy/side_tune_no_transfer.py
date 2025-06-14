import argparse

import time
import os
import json
from loader import MoleculeDataset
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import GNN
from torch_geometric.nn import global_mean_pool
from sklearn.metrics import roc_auc_score,auc,precision_recall_curve

from tqdm import tqdm
import numpy as np

from model_sider import SideMLP, GNN_graphpred_side, GNN_graphpred_no_transfer
from sklearn.metrics import roc_auc_score

from splitters import scaffold_split, random_split, random_scaffold_split
import pandas as pd

criterion = nn.BCEWithLogitsLoss(reduction = "none")

def train(args, model, device, loader, optimizer):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        # print('batch')
        pred = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        y = torch.nn.functional.one_hot(batch.y, num_classes=2).to(torch.float32)
        #y = batch.y.view(pred.shape).to(torch.float64)
        #y = batch.y.to(torch.float64)

        #Whether y is non-null or not.
        #is_valid = y**2 > 0
        is_valid = (y == 0) | (y == 1)
        #Loss matrix
        #loss_mat = criterion(pred.double(), (y+1)/2)
        loss_mat = criterion(pred.double(), y)
        #loss matrix after removing null target
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
            
        optimizer.zero_grad()
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        loss.backward()

        optimizer.step()


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

    roc_list, pr_list = [], []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == 0) > 0:
            #is_valid = y_true[:,i]**2 > 0
            #roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))
            is_valid = (y_true[:, i] == 0) | (y_true[:, i] == 1)
            roc_list.append(roc_auc_score(y_true[is_valid, i], y_scores[is_valid, i]))
            precision, recall, thresholds = precision_recall_curve(y_true[is_valid, i], y_scores[is_valid, i])
            pr_list.append(auc(recall, precision))
    
    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list), sum(pr_list)/len(pr_list)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=1,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--dataset', type=str, default = 'cora', help='root directory of dataset. For now, only classification.')
    parser.add_argument('--input_model_file', type=str, default = 'models_graphcl/graphcl_cora_final.pth', help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default = '', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--split', type = str, default="scaffold", help = "random or scaffold or random_scaffold")
    parser.add_argument('--eval_train', type=int, default = 1, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 4, help='number of workers for dataset loading')
    parser.add_argument('--source', type=str, default = 'cora', help='root directory of dataset. For now, only classification.')
    args = parser.parse_args()


    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)

    #Bunch of classification tasks
    if args.dataset == "tox21":
        num_tasks = 12
    elif args.dataset == "hiv":
        num_tasks = 1
    elif args.dataset == "pcba":
        num_tasks = 128
    elif args.dataset == "muv":
        num_tasks = 17
    elif args.dataset == "bace":
        num_tasks = 1
    elif args.dataset == "bbbp":
        num_tasks = 1
    elif args.dataset == "toxcast":
        num_tasks = 617
    elif args.dataset == "sider":
        num_tasks = 27
    elif args.dataset == "clintox":
        num_tasks = 2
    else:
        num_tasks = 2
        #raise ValueError("Invalid dataset name.")

    #set up dataset
    #dataset = MoleculeDataset("dataset/" + args.dataset, dataset=args.dataset)

    #print(dataset)
 
    #if args.split == "scaffold":
    #    print("scaffold")
    #    smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    #    train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1)
    #    print("scaffold")
    #elif args.split == "random":
    #    train_dataset, valid_dataset, test_dataset = random_split(dataset, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    #    print("random")
    #elif args.split == "random_scaffold":
    #    smiles_list = pd.read_csv('dataset/' + args.dataset + '/processed/smiles.csv', header=None)[0].tolist()
    #    train_dataset, valid_dataset, test_dataset = random_scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,frac_valid=0.1, frac_test=0.1, seed = args.seed)
    #    print("random scaffold")
    #else:
    #    raise ValueError("Invalid split option.")

    #print(train_dataset[0])

    PATH = f'''{os.getcwd()}/datasets/'''.replace('easy', 'data_processing')
    train_dataset = MoleculeDataset(PATH + args.source, dataset=args.dataset, fold=3)
    valid_dataset = MoleculeDataset(PATH + args.source, dataset=args.dataset, fold=2)
    test_dataset = MoleculeDataset(PATH + args.source, dataset=args.dataset, fold=1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    #set up model
    model = GNN_graphpred_no_transfer(args.num_layer, args.emb_dim, num_tasks, JK = args.JK, drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling)
    model.to(device)
    print(model)

    #set up optimizer
    #different learning rate for different part of GNN
    model_param_group = []
    for i in model.named_parameters():
        print(i)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.decay)
    print(optimizer)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []

    train_pr_list = []
    val_pr_list = []
    test_pr_list = []

    train_acc = 0
    train_pr = 0
    val_acc, val_pr = 0,0
    test_acc, test_pr = 0,0

    print('side tuning ...')
    start = time.time()
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch))
        
        train(args, model, device, train_loader, optimizer)

        print("====Evaluation")
        if args.eval_train:
            train_acc, train_pr = eval(args, model, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
            train_pr = 0
        val_acc, val_pr = eval(args, model, device, val_loader)
        test_acc, test_pr = eval(args, model, device, test_loader)

        print("ROC: train: %f val: %f test: %f" %(train_acc, val_acc, test_acc))
        print("PR: train: %f val: %f test: %f" %(train_pr, val_pr, test_pr))

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        val_pr_list.append(val_pr)
        test_pr_list.append(test_pr)
        train_pr_list.append(train_pr)

        print("")

    end = time.time()
    
    logits = eval(args, model, device, test_loader, only_pred=True)
    predictions = [torch.sigmoid(tensor).cpu().numpy().tolist() for tensor in logits]  # Bring me some probabilities!

    val_acc, val_pr = eval(args, model, device, val_loader)
    test_acc, test_pr = eval(args, model, device, test_loader)

    # Save to a JSON file
    path = os.getcwd()
    try:
        os.mkdir(path + "/probabilities")
    except:
        print("probabilities folder already created")

    with open(path + f"/probabilities/{args.dataset}_probabilities.json", "w") as json_file:
        json.dump(predictions, json_file)

    try:
        os.mkdir('outputs')
    except:
        pass
    
    with open(f'outputs/{args.dataset}_regular_result.log', 'a+') as f:
        f.write(args.dataset + ' ' + str(args.runseed) + ' ROC Train ' + str(train_acc))
        f.write(args.dataset + ' ' + str(args.runseed) + ' ROC Val ' + str(val_acc))
        f.write(args.dataset + ' ' + str(args.runseed) + ' ROC Test ' + str(test_acc))
        f.write('\n')
        f.write(args.dataset + ' ' + str(args.runseed) + ' PR Train ' + str(train_pr))
        f.write(args.dataset + ' ' + str(args.runseed) + ' PR Val ' + str(val_pr))
        f.write(args.dataset + ' ' + str(args.runseed) + ' PR Test ' + str(test_pr))
        f.write('\n')
        f.write('Time: ' + str(end-start))
        f.write('\n')

if __name__ == "__main__":
    main()
