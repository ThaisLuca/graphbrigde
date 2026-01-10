import json
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

    data_to_json = [
        [
            {
                target_name: [],
                "atomicNumber": [],
                "chiralityNumber": [],
                "bond": [],
                "stereo": []
            }
        ],
        [
            {
                target_name: [],
                "atomicNumber": [],
                "chiralityNumber": [],
                "bond": [],
                "stereo": []
            }
        ]
    ]
    
    toxicity = False
    if dataset in ["clintox_toxicity", "clintox_approve"]:
        if "toxicity" in dataset:
            toxicity = True
        dataset = "clintox"

    dataset = MoleculeDataset("dataset/" + dataset, dataset=dataset)
    print(f"{dataset} has {len(dataset)} instances")
    
    for i, data in enumerate(dataset):

        molecule_name = data_name + str(i+1)

        # Now identify if it is a positive/negative example!
        target = data.y.tolist()[0]

        if target == -1:
            neg.append(
                f"{target_name}({molecule_name})."
            )

            data_to_json[1][0][target_name].append([molecule_name])
        else:
            pos.append(
                f"{target_name}({molecule_name})."
            )

            data_to_json[0][0][target_name].append([molecule_name])

        for item in range(0, data.x.shape[0]):
            atomic_number, chirality_number = data.x[item].tolist()
            
            node_type = f"node{item}"

            # All datasets have two features
            # Every feature becomes a fact
            #   [[5, 0], -> feature1(molecule1, node1, 5), feature2(molecule1, node1, 0)
            #    [6,0]]  -> feature1(molecule1, node2, 6), feature2(molecule1, node2, 0)
            if target != -1:
            
                facts.append(
                    f"atomicNumber({molecule_name},{node_type},{atomic_number})."
                )

                facts.append(
                    f"chiralityNumber({molecule_name},{node_type},{chirality_number})."
                )

            data_to_json[0][0]["atomicNumber"].append([molecule_name,node_type,atomic_number])
            data_to_json[0][0]["chiralityNumber"].append([molecule_name,node_type,chirality_number])
        
        for idx in range(data.num_edges):

            src = data.edge_index[0, idx]
            dst = data.edge_index[1, idx]

            #src_atomic_number, src_chirality_number = data.x[src].tolist()
            #dst_atomic_number, dst_chirality_number = data.x[dst].tolist()

            src_node_type, dst_node_type = f"node{src}", f"node{dst}"

            # Get edge attribute
            bondtype, stereo = data.edge_attr[idx].tolist()

            # Every edge becomes a fact too
            #   [[5, 0]] -> bond(molecule1, node1, node2, 5), stereo(molecule1, node1, node2, 0)

            if target != -1:
                facts.append(
                    f"bond({molecule_name},{src_node_type},{dst_node_type},{bondtype})."
                )

                facts.append(
                    f"stereo({molecule_name},{src_node_type},{dst_node_type},{stereo})."
                )

            data_to_json[0][0]["bond"].append([molecule_name,src_node_type,dst_node_type,bondtype])
            data_to_json[0][0]["stereo"].append([molecule_name,src_node_type,dst_node_type,stereo])

    json.dump(data_to_json, open(f"{data_name}.json", "w"))
    
    facts,pos,neg = list(set(facts)), list(set(pos)), list(set(neg))
    facts.sort()
    return facts,pos,neg


if __name__ == "__main__":
     
     for dataset in datasets_to_go_relational:
        PATH = f"dataset/{dataset}/processed/geometric_data_processed_{FOLD}.pt"
        facts,pos,neg = convert(path=PATH, dataset=dataset, check_sanity=False)

        write_to_file("bk.pl", facts)
        write_to_file("pos.pl", pos)
        write_to_file("neg.pl", neg)