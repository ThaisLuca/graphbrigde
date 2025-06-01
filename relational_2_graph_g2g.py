import re
import os
import torch
import random
import subprocess
import numpy as np
import networkx as nx
from tqdm import tqdm
from os import makedirs, getcwd
from torch_geometric.data import Data
import torch_geometric.utils as tutils
from torch_geometric.data import InMemoryDataset
from torch_geometric.transforms import SVDFeatureReduction

import sys
sys.path.append('.')
from get_datasets import datasets

from experiments import bk, experiments, sizes


relations, constants_and_relations = [],{}

def load_file(filename):
    with open(filename, 'r') as file:
        return file.read()
    
def create_subhgraph(facts, label, fold, attributes, e_attributes, id=1):
    relations, nodes, edges, numbers, constants = [],[],[],[],[]

    for lit in facts:

        relation = re.findall(r'([a-z\d_]+)\(', lit)[0]
        data = re.findall(r'\((.*?)\)', lit)

        relations.append(relation)
        ent1,ent2 = data[0].split(",")
        edges.append((ent1,ent2,relation))

        for nd in data[0].split(","):

            if len(nd.strip()) == 0: 
                continue

            #if nd.isdigit():
            #    numbers.append(nd)
            #else:
            #    constants.append(nd)
            
            if nd not in nodes:
                nodes.append(nd)

    #relations, constants, numbers = list(set(relations)), list(constants_and_relations.keys()), list(set(numbers))
    #attributes = sorted(constants) + numbers

    mapping_e = dict(zip(list(set(relations)), range(len(list(set(relations))))))
    mapping_v = dict(zip(list(set(nodes)), range(len(list(set(nodes))))))

    # Map nodes and its attributes
    x = [0]*len(nodes)
    for i, node in enumerate(nodes):
        
        aux = np.zeros((len(attributes)))
        aux[np.where(np.array(attributes) == node)] = 1
        
        x[mapping_v[node]] = aux.copy()

    # Map edges and its attributes
    edge_attr = torch.empty((len(edges), len(e_attributes)), dtype=torch.long) 
    edge_index = torch.empty((2, len(edges)), dtype=torch.long)
    for i, (src, dst,relation) in enumerate(edges):
        edge_index[0, i] = mapping_v[src]
        edge_index[1, i] = mapping_v[dst]

        aux = np.zeros((len(e_attributes)))
        aux[mapping_e[relation]] = 1
        edge_attr[i] = torch.tensor(aux.copy(), dtype=torch.long)

    data = Data(x=torch.tensor(np.array(x), dtype=torch.long), 
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor(np.array([label]), dtype=torch.long),
                fold=torch.tensor(np.array([fold]), dtype=torch.long),
                id=torch.tensor(np.array([id]), dtype=torch.long))
    return data

def build_dataset(literals, constants_and_relations, relations, fold):
    print("Creating dataset...")

    graph_list = []
    targets = []

    attributes = list(constants_and_relations.keys())
    e_attributes = list(set(relations))

    for tgt,pin in enumerate(["neg", "pos"]):
        for i,dt in enumerate(literals[pin]):

            relation = re.findall(r'([a-z\d_]+)\(', dt)[0]
            data = re.findall(r'\((.*?)\)', dt)

            facts = []
            for nd in data[0].split(","):

                if nd in constants_and_relations:
                    facts += constants_and_relations[nd]

            if facts:
                graph_list.append(create_subhgraph(facts, tgt, fold, attributes, e_attributes, i))
                targets.append(tgt)
    return graph_list

def map_constants_and_relations(facts):
    constants_relations = {}
    relations = []
    for dt in facts:

        relation = re.findall(r'([a-z\d_]+)\(', dt)[0]
        data = re.findall(r'\((.*?)\)', dt)

        if relation not in relations:
            relations.append(relation)

        for nd in data[0].split(","):
            
            if nd not in constants_relations:
                constants_relations[nd] = []
            
            constants_relations[nd].append(dt)
    return constants_relations, relations

class MoleculeDataset_aug(InMemoryDataset):
    def __init__(self,
                 root,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset='zinc250k',
                 empty=False,
                 aug="none", aug_ratio=None):
        """
        Adapted from qm9.py. Disabled the download functionality
        :param root: directory of the dataset, containing a raw and processed
        dir. The raw dir should contain the file containing the smiles, and the
        processed dir can either empty or a previously processed file
        :param dataset: name of the dataset. Currently only implemented for
        zinc250k, chembl_with_labels, tox21, hiv, bace, bbbp, clintox, esol,
        freesolv, lipophilicity, muv, pcba, sider, toxcast
        :param empty: if True, then will not load any data obj. For
        initializing empty dataset
        """
        self.dataset = dataset
        self.root = root
        self.aug = aug
        self.aug_ratio = aug_ratio

        super(MoleculeDataset_aug, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self._data, self.slices = torch.load(self.processed_paths[0],weights_only=False)

    def get(self, idx):
        data = Data()
        for key in self._data.keys():
            item, slices = self._data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]

        if self.aug == 'dropN':
            data = drop_nodes(data, self.aug_ratio)
        elif self.aug == 'permE':
            data = permute_edges(data, self.aug_ratio)
        elif self.aug == 'maskN':
            data = mask_nodes(data, self.aug_ratio)
        elif self.aug == 'subgraph':
            data = subgraph(data, self.aug_ratio)
        elif self.aug == 'random':
            n = np.random.randint(2)
            if n == 0:
                data = drop_nodes(data, self.aug_ratio)
            elif n == 1:
                data = subgraph(data, self.aug_ratio)
                # data = subgraph(data, 0.5)
            else:
                print('augmentation error')
                assert False
        elif self.aug == 'none':
            None
        else:
            print('augmentation error')
            assert False

        return data


    @property
    def raw_file_names(self):
        try:
            file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        except:
            file_name_list = []
        return file_name_list

    @property
    def processed_file_names(self):
        return 'geometric_data_processed.pt'

    def download(self):
        pass
        #raise NotImplementedError('Must indicate valid location of raw data. '
                                  #'No download allowed')

    def process(self):
        
        # Load source dataset
        src_total_data = datasets.load(source, bk[source], seed=441773)
        src_data = datasets.load(source, bk[source], target=predicate, balanced=False, seed=441773)

        # Group folds to get relations and constants
        src_facts = datasets.group_folds(src_data[0])
        src_facts = [rel for rel in src_facts if 'recursion' not in rel]
        constants_and_relations, relations = map_constants_and_relations(src_facts)

        # Split folds into facts, pos and neg
        src_facts = src_data[0]
        src_pos   = src_data[1]
        src_neg   = src_data[2]

        if source in ['nell_sports', 'nell_finances', 'yago2s']:
            n_folds = 3
        else:
            n_folds = len(src_facts)


        graph_list = []
        for i in range(n_folds):
            print('\n Starting fold {} of {} folds \n'.format(i+1, n_folds))

            if source in ['nell_sports', 'nell_finances', 'yago2s']:
                #[src_train_facts, src_test_facts] =  [src_data[0][0], src_data[0][0]]
                to_folds_pos = datasets.split_into_folds(src_data[1][0], n_folds=n_folds, seed=441773)
                to_folds_neg = datasets.split_into_folds(src_data[2][0], n_folds=n_folds, seed=441773)
                [src_train_pos, src_test_pos] =  datasets.get_kfold_small(i, to_folds_pos)
                [src_train_neg, src_test_neg] =  datasets.get_kfold_small(i, to_folds_neg)

                src_pos_fold = src_train_pos + src_test_pos
                src_neg_fold = src_train_neg + src_test_neg

            else:
                #src_facts_fold = [rel for rel in src_facts[i] if 'recursion' not in rel] #filter relations that use recursion
                src_pos_fold = src_pos[i]
                src_neg_fold = src_neg[i]

            #constants_and_relations, relations = map_constants_and_relations(src_facts_fold)

            #print(constants_and_relations)
        
            curr_graph_list = build_dataset({"pos": src_pos_fold, "neg": src_neg_fold}, constants_and_relations, relations, i)
            print(len(curr_graph_list))

            for idx,item in enumerate(curr_graph_list):
                graph_list.append(item)
                    
            # Convert to InMemoryDataset format
            data, slices = self.collate(curr_graph_list)
            torch.save((data, slices), f'''datasets/{source}/processed/geometric_data_processed_{i+1}.pt''')
        
        data, slices = self.collate(graph_list)
        print(len(graph_list))
        torch.save((data, slices), f'''datasets/{source}/processed/geometric_data_processed_full.pt''')
        #torch.save(graph_list, f'''dataset/{source}/{source}_data_full.pt''')



try:
    os.mkdir('dataset')
except:
    pass

for experiment in experiments:

    source = experiment['source']
    predicate = experiment['predicate']

    try:
        os.mkdir(f'''dataset/{source}''')
    except:
        pass

    print('Running experiment for %s' % source)

    PATH = f'''{os.getcwd()}/datasets/'''
    if not os.path.isdir(PATH):
        os.makedirs(PATH)

    filename = PATH + '/' + source

    MoleculeDataset_aug(filename)

    