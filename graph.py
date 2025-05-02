import re
import json
import operator
import networkx as nx
import matplotlib.pyplot as plt
from experiments import experiments, bk
from torch_geometric.utils.convert import from_networkx

import sys
sys.path.append('.')
from data_utils.get_datasets import datasets
from sklearn.preprocessing import LabelBinarizer
#from utils import write_examples


class Graph:
    def __init__(self,
                 data_dir,
                 predicate):
                 #log_dir,
                 #params,
                 #n_splits,
                 #n_samples,
                 #dedup=False,
                 #cached=True,
                 #use_gpu=True,
                 #no_logger=False,
                 #progress_bar=False):
        self.data_dir = data_dir
        self.predicate = predicate
        '''self.cached = cached
        self.log_dir = log_dir
        self.params = params
        self.n_splits = n_splits
        self.n_samples = n_samples
        self.dedup = dedup
        # self.device = 'cuda:0' if use_gpu else 'cpu'
        self.use_gpu = use_gpu
        self.progress_bar = progress_bar'''

        self.X = None
        self.y = None

        self.network = None
        self.datasets = datasets()
    
    def init_data(self):

        # Positive Examples
        self.pos = True
        self.network = self.create_graph()
        #communities = self.generate_communities()
        #self.get_most_representative(communities)

        # Negative Examples
        #self.pos = False
        #self.network = self.create_graph()
        #communities = self.generate_communities()
        #self.get_most_representative(communities)
        
    def create_graph(self, directed=False, colors=None):
        '''
            Creates a new graph based on data passed as parameter
            Args:
                data: information to build the graph
                directed: True if the graph is directed; False if it is not
            Returns:
                graph object
        '''
        self.pos = True
        nodes = set()
        count = 0
        # movies, accounts, sportsteam, venues and titles are yellow
        # genre, authors, locations and levels are purple
        # persons, words, sportsleague, classes and titles are blue
        # courses, company and proteins are darkblue

        #colors = {'movie': [0.8,0.1], 'teamplaysinleague': [0.8,0.1], 'teamalsoknownas': [0.8,0.8], 'venue': [0.1,0.8], 'follows': [0.8,0.8], 'tweets': [0.8,0.1], 'publication': [0.8,0.1], 'actor': [0.2,0.2], 'director': [0.3,0.3], 'genre': [0.1,0.5], 'author': [0.1,0.5], 'interaction': [0.3,0.3], 'location': [0.3,0.5], 'courselevel': [0.3,0.5], 'acquired': [0.3,0.3], 'companyalsoknownas': [0.3,0.3], 'female': [0.8,0.8]}

        #print("Creating graph...")
        g = nx.Graph()
        lb = LabelBinarizer()

        source = self.data_dir.split('/')[1]
        
        src_total_data = self.datasets.load(source, bk[source], seed=441773)
        src_data  = self.datasets.load(source, bk[source], target=self.predicate, balanced=False, seed=441773)
        
        if self.pos:
            #print("Using positive examples.")
            data = self.datasets.group_folds(src_data[0]) + self.datasets.group_folds(src_data[1])
        #else:
            #print("Using negative examples.")
            #data = self.datasets.group_folds(src_data[0]) + self.datasets.group_folds(src_data[2])

        nodes, edges, nodes_2_binarize, edges_2_binarize, curr_edges = [],[],[],[],[]
        node_mapping, edge_mapping = {},{}
        for dt in data:

            if 'recursion' in dt or 'ta' in dt:
                continue

            # Check if it is binary relation
            if ',' in dt:
                from_node, to_node = dt.replace(').','').split('(')[1].split(',')
                relation = dt.replace(').','').split('(')[0]

                if from_node not in node_mapping:
                    node_mapping[from_node] = len(node_mapping)
                    nodes_2_binarize.append(from_node)

                if to_node not in node_mapping:
                    node_mapping[to_node] = len(node_mapping)
                    nodes_2_binarize.append(to_node)

                if relation not in edge_mapping:
                    edge_mapping[relation] = len(edge_mapping)
                    edges_2_binarize.append(relation)

                curr_edges.append((node_mapping[from_node], node_mapping[to_node], relation))
            else:
                print(source, dt)
                return
            
        # Map string node names to integer indices
        #node_mapping = {name: idx for idx, name in enumerate(nodes)}
        #G = nx.relabel_nodes(g, node_mapping)
        #(2, {"name": "Bob", "age": 30, "city": "London"})
        
        lb.fit(nodes_2_binarize)
        encoding_labels = lb.transform(nodes_2_binarize)

        nodes = [(node_mapping[name], {"feature": encoding_labels[node_mapping[name]]}) for name in node_mapping]
        g.add_nodes_from(nodes)

        lb.fit(edges_2_binarize)
        encoding_labels = lb.transform(edges_2_binarize)

        edges = [(e[0],e[1],{"feature": encoding_labels[edge_mapping[e[2]]]}) for e in curr_edges]

        g.add_edges_from(edges)

        pyg_graph = from_networkx(g)
        #print(g.nodes[0]["name"])
        #pyg_graph.x = torch.tensor([g.nodes[n]["name"] for n in g.nodes()], dtype=torch.float)

        print("Graph successfully created.")
        print("  It contains %d vertices and %d edges." % (g.number_of_nodes(), g.number_of_edges()))
        print("  It contains %d connected components." % (nx.number_connected_components(g)))

        #[print(node, "has self loop") for node in g.nodes() if self.has_self_loop(g, node) == True]

        return pyg_graph, node_mapping, g #, relations_of_nodes, count_relations
    
    def has_self_loop(self, g, node):
        try:
            if g[node][node] != None:
                return True
        except Exception:
            return False

    def generate_communities(self):
        print("Generating communities...")        
        communities = list(nx.community.louvain_communities(self.network))
        print("  It contains %d communities." % (len(communities)))
        #nodes = communities[1].union(communities[2]).union(communities[0])
        #nx.draw(self.network.subgraph(list(nodes)), with_labels=True)
        #print(nx.get_edge_attributes(self.network,'r_name'))
        
        #n_vertices_communities = [len(c) for c in communities]
        #print("  Communities contain the following number of vertices: ", n_vertices_communities)
        #n_vertices_count_list = [list.count(n_vertices_communities, i) for i in n_vertices_communities]
        
        return communities

    def get_most_representative(self, communities, metric='betweenness'):
        examples = []
        for cmt in communities:
            G = self.network.subgraph(list(cmt))
            bts = nx.edge_betweenness_centrality(G, normalized=True)
            bts_sorted = self.make_it_pretty(sorted(bts.items(), key=operator.itemgetter(1)))
            edges_data = [(edge[0], edge[1], self.network.get_edge_data(edge[0], edge[1])['r_name']) for edge in bts_sorted]
            examples += self.filter_edges_by_name(edges_data)[:self.n_samples]

        file_name=self.data_dir + '/pos.pl' if self.pos else self.data_dir + '/neg.pl'
        write_examples(self.build_examples(examples), file_name)

    def build_examples(self, data):
        return [f'{d[2]}({d[0]},{d[1]})' for d in data]

    def filter_edges_by_name(self, data):
        return [d for d in data if d[2] == self.predicate]

    def make_it_pretty(self, data):
        #(('asciveresmarianna', 'asaramay'), 2.7132624267419144e-05)
        return [(d[0][0], d[0][1], d[1]) for d in data]

    # function to create node colour list
    def create_community_node_colors(self, communities):
        number_of_colors = len(communities)
        colors = ["#D4FCB1", "#CDC5FC", "#FFC2C4", "#F2D140", "#BCC6C8"]*number_of_colors
        colors = colors[:number_of_colors]
        node_colors = []
        for node in self.network:
            current_community_index = 0
            for community in communities:
                if node in community:
                    node_colors.append(colors[current_community_index])
                    break
                current_community_index += 1
        return node_colors

    def plot_communities(self, communities, range_list):
        fig, ax = plt.subplots(3, figsize=(15, 20))

        # Plot graph with colouring based on communities
        for i, c in enumerate(range_list):
            self.visualize_communities(communities[c], i+1)
        plt.show()

    # function to plot graph with node colouring based on communities
    def visualize_communities(self, communities, i):
        node_colors = self.create_community_node_colors(communities)
        modularity = round(nx.community.modularity(self.network, communities), 6)
        title = f"Community Visualization of {len(communities)} communities with modularity of {modularity}"
        pos = nx.spring_layout(self.network, k=0.3, iterations=50, seed=2)
        plt.subplot(3, 1, i)
        plt.title(title)
        nx.draw(
            self.network,
            pos=pos,
            node_size=1000,
            node_color=node_colors,
            with_labels=True,
            font_size=20,
            font_color="black",
        )       