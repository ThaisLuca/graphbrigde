
import torch
import networkx as nx
import numpy as np

from collections import Counter
from easy.loader import MoleculeDataset
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_networkx, from_networkx

FOLD = 1

datasets_to_go_relational = {"bace":    # molecule is an active BACE-1 inhibitor (binary label: active/inactive)
                               {"target": "inhibitor"},
                            #"bbbp": # Blood-brain barrier penetration (BBBP) dataset comes modeling and prediction of the barrier permeability (binary label: permeable/impermeable)
                            #    {"target": "permeability"},
                            #"hiv": 
                            #    {"target": "active"},
                            #"clintox_toxicity": # clinical trial toxicity (or absence of toxicity)
                            #    {"target": "toxicity"},
                            #"clintox_approve": # FDA approval status
                            #    {"target": "approved"},
                            #"zinc_standard_agent": # identify structures that are likely to bind to drug targets
                            #    {"target": []}
                            }


def write_to_file(filename, data):
    with open(filename, "w") as f:
        for dt in data:
            f.write(dt)
            f.write("\n")

def convert(path, dataset, check_sanity=False):
    facts, pos, neg = [], [], []

    print(dataset)

    data_name = dataset
    target_name = datasets_to_go_relational[dataset]["target"]
    
    toxicity = False
    if dataset in ["clintox_toxicity", "clintox_approve"]:
        if "toxicity" in dataset:
            toxicity = True
        dataset = "clintox"

    dataset = MoleculeDataset("dataset/" + dataset, dataset=dataset)
    print(f"{dataset} has {len(dataset)} instances")
    
    node_type_mapping, edge_type_mapping = {}, {}
    for i, data in enumerate(dataset):

        molecule_name = data_name + str(i+1)

        # Now identify if it is a positive example!
        target = data.y.tolist()[0]

        if target == -1:
            neg.append(
                f"{target_name}({molecule_name})."
            )
        else:
            pos.append(
                f"{target_name}({molecule_name})."
            )

        # TODO: separar apenas um fold. O que tiver mais samples

        for item in range(0, data.x.shape[0]):
            atomic_number, chirality_number = data.x[item].tolist()
            
            if atomic_number not in node_type_mapping:
                node_type_mapping[atomic_number] = f"node{atomic_number}"
            node_type = node_type_mapping[atomic_number]

            # All datasets have two features
            # Every feature becomes a fact
            #   [[5, 0], -> feature1(molecule1, node1, 5), feature2(molecule1, node1, 0)
            #    [6,0]]  -> feature1(molecule1, node2, 6), feature2(molecule1, node2, 0)
            facts.append(
                f"atomicNumber({molecule_name},{node_type},{atomic_number})."
            )

            facts.append(
                f"chiralityNumber({molecule_name},{node_type},{chirality_number})."
            )


        for idx, (src, dst) in enumerate(data.edge_index.t().tolist()):

            src_atomic_number = data.x[src].tolist()[0]
            dst_atomic_number = data.x[dst].tolist()[0]

            src_node_type, dst_node_type = node_type_mapping[src_atomic_number], node_type_mapping[dst_atomic_number]

            # Get edge attribute
            bondtype, stereo = data.edge_attr[idx].tolist()

            # Every edge becomes a fact too
            #   [[5, 0]] -> bond(molecule1, node1, node2, 5), stereo(molecule1, node1, node2, 0)
            facts.append(
                f"bond({molecule_name},{src_node_type},{dst_node_type},{bondtype})."
            )

            facts.append(
                f"stereo({molecule_name},{src_node_type},{dst_node_type},{stereo})."
            )
    
    return facts,pos,neg


if __name__ == "__main__":
     
     for dataset in datasets_to_go_relational:
        PATH = f"dataset/{dataset}/processed/geometric_data_processed_{FOLD}.pt"
        facts,pos,neg = convert(path=PATH, dataset=dataset, check_sanity=False)

        write_to_file("bk.pl", set(facts))
        write_to_file("pos.pl", set(pos))
        write_to_file("neg.pl", set(neg))
