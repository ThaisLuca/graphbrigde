

from torch_geometric.datasets import ZINC

dataset = ZINC(root=".")



#import deepchem as dc

#dataset_dc = dc.molnet.load_zinc15(data_dir="data/",save_dir="data_c/",dataset_size="250K",dataset_dimension="2D")

#x,y,w,ids = train.X, train.y, train.w, train.ids