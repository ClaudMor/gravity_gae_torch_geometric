import os

os.chdir("./gravity_gae")

import scipy.sparse as sp
import torch
from torch import tensor as tt
from torch.nn import  Sequential, ReLU, BCEWithLogitsLoss
from torch_geometric import seed_everything
from torch_geometric.data import Data
import torch_geometric.transforms as T
from torch.optim.lr_scheduler import ReduceLROnPlateau


import methods
os.chdir("..")


seed_everything(12345)             # Seed
use_sparse_representation = True   # Use sparse matrices to perform messgae passing (https://pytorch-geometric.readthedocs.io/en/latest/advanced/sparse_tensor.html)
lrscheduler = None                 # E.g. ReduceLROnPlateau(optimizer, "min", factor = 0.5, patience = 20, verbose = True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')       # One of `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` or `torch.device("cpu")`
hidden_dimension = 64              # Hidden dimentsion for first layer
output_dimension = 32              # Output dimension
num_epochs = 200                   # Number of epochs
num_runs = 5                       # Number of initial configuration to average over
lr = 0.05                          # Learning rate



# Load data (use same functions as tensorflow's implementation)
adj_init, features = methods.load_data("cora", "./data/cora.cites" )
adj, val_edges, val_edges_false, test_edges, test_edges_false =  methods.mask_test_edges_general_link_prediction(adj_init, 10., 5.)

# Convert scipy matrices to torch_geometric's Data
features = torch.tensor(features.todense(), dtype = torch.float32)
train_dense_adjm = torch.tensor((adj + sp.eye(adj.shape[0])).todense())
edge_label_train_general = train_dense_adjm.reshape(-1)
train_edge_index = torch.tensor(methods.sparse_to_tuple(adj + sp.eye(adj.shape[0]))[0], dtype = torch.int64).t()

train_data = Data( x = features, edge_index = train_edge_index, edge_label = edge_label_train_general, edge_label_index = "salha_general")

val_data = Data( x = features, edge_index = torch.tensor(methods.sparse_to_tuple(adj + sp.eye(adj.shape[0]))[0], dtype = torch.int64).t(), edge_label_index = torch.cat((tt(val_edges), tt(val_edges_false)), dim = 0).t(), edge_label = torch.cat((torch.ones(val_edges.shape[0]), torch.zeros(val_edges_false.shape[0]))))

test_data = Data( x = features, edge_index = torch.tensor(methods.sparse_to_tuple(adj + sp.eye(adj.shape[0]))[0], dtype = torch.int64).t(), edge_label_index = torch.cat((tt(test_edges), tt(test_edges_false)), dim = 0).t(), edge_label = torch.cat((torch.ones(test_edges.shape[0]), torch.zeros(test_edges_false.shape[0]))))


# us sparse representation and/or GPU, if required
if use_sparse_representation:
    tosparse = T.ToSparseTensor()
    train_data = tosparse(train_data)
    val_data = tosparse(val_data)
    test_data = tosparse(test_data).cpu()

train_data.to(device, "x", "adj_t", "edge_label")
val_data.to(device,"x", "adj_t","edge_label_index", "edge_label")


# Get input dimension
input_dimension = train_data.x.shape[1]



# Model definition
unwrapped_layers_kwargs = [
                    {"layer":methods.Conv(input_dimension, hidden_dimension), 
                    "normalization_before_activation": None, 
                    "activation": ReLU(), 
                    "normalization_after_activation": None, 
                    "dropout_p": None, 
                    "_add_remaining_self_loops": False, 
                    "uses_sparse_representation": use_sparse_representation,
                    },

                    {"layer":methods.Conv(hidden_dimension, output_dimension + 1), 
                    "normalization_before_activation": None, 
                    "activation": None, 
                    "normalization_after_activation": None, 
                    "dropout_p": None, 
                    "_add_remaining_self_loops": False, 
                    "uses_sparse_representation": use_sparse_representation,
                    },
                    {"layer":methods.Conv(hidden_dimension, output_dimension + 1), 
                    "normalization_before_activation": None, 
                    "activation": None, 
                    "normalization_after_activation": None, 
                    "dropout_p": None, 
                    "_add_remaining_self_loops": False, 
                    "uses_sparse_representation": use_sparse_representation,
                    }]







encoder = methods.GNN_FB(gnn_layers = [ methods.LayerWrapper(**unwrapped_layers_kwargs[0]),
                                    methods.ParallelLayerWrapper([methods.LayerWrapper(**unwrapped_layers_kwargs[1]), methods.LayerWrapper(**unwrapped_layers_kwargs[2])]),
                                    methods.VGAE_Reparametrization()])

decoder = methods.LinkPropertyPredictorGravity(l = 1., CLAMP = 4)

model = Sequential(encoder, decoder).to(device)


optimizer = torch.optim.Adam(model.parameters(), lr = lr) 


# Array of test set performances
perfs = []

# Compute train and validation set loss normalization
tot_train_edges = train_data.edge_label.size(0)
tot_pos_train_edges = int(train_data.edge_label.sum())
tot_neg_edges_train = tot_train_edges - tot_pos_train_edges
pos_weight = torch.tensor(tot_neg_edges_train / tot_pos_train_edges) # 5
norm = tot_train_edges / (2 * tot_neg_edges_train) 



tot_val_edges = val_data.edge_label.size(0)
tot_pos_val_edges = int(val_data.edge_label.sum())
tot_neg_edges_val = tot_val_edges - tot_pos_val_edges
val_pos_weight = torch.tensor(neg_sampling_ratio_val)
norm_val =  tot_val_edges / (2 * tot_neg_edges_val) 

for i in range(num_runs):
    methods.train(train_data, model,  methods.VGAELossWrapper(norm,BCEWithLogitsLoss(pos_weight = pos_weight)), optimizer, device, num_epochs, lrscheduler = lrscheduler, early_stopping = True, val_data = val_data,  val_loss_fn =  methods.losses_sum_closure([methods.auc_loss, methods.ap_loss])  , patience = 50, retrain_data = train_data, use_sparse_representation = use_sparse_representation,  epoch_print_freq = 10  )

    perfs.append(methods.evaluate_link_prediction(model, test_data, None, device = device))

# Print test-set performances
mean_std_dict = methods.summarize_link_prediction_evaluation(perfs)
markdown_table = methods.pretty_print_link_performance_evaluation(mean_std_dict, "Gravity-VGAE")
print(markdown_table)


# |       | Gravity-VGAE   |
# |:------|:---------------|
# | AUC   | 0.901 +- 0.004 |
# | F1    | 0.742 +- 0.01  |
# | hitsk | 0.75 +- 0.02   |
# | AP    | 0.912 +- 0.007 |