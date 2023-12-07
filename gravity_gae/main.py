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

# Please set the parameters below
seed_everything(12345)             # Seed
model_name = "gravity_gae"              # Please specify what model you'd liek to use. Must be one of {"gravity_gae", }
dataset = "cora"     # Only "cora" is implemented right now
task    = "biased"     # One of "general", "biased", "bidirectional"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')      # One of `torch.device('cuda' if torch.cuda.is_available() else 'cpu')` or `torch.device("cpu")`
num_runs = 1            # Number of initial configuration to average over
lrscheduler = None
lr                        = methods.setup_suggested_parameters_sets[dataset][task][model_name]["lr"]
num_epochs                = methods.setup_suggested_parameters_sets[dataset][task][model_name]["num_epochs"]
val_loss_fn               = methods.setup_suggested_parameters_sets[dataset][task][model_name]["val_loss_fn"]
early_stopping            = methods.setup_suggested_parameters_sets[dataset][task][model_name]["early_stopping"]
use_sparse_representation = methods.models_suggested_parameters_sets[dataset][task][model_name]["use_sparse_representation"]
####


model = None
if model_name == "gravity_gae":
    model = methods.get_gravity_gae(**methods.models_suggested_parameters_sets[dataset][task][model_name])
elif model_name == "gravity_vgae":
    model = methods.get_gravity_vgae(**methods.models_suggested_parameters_sets[dataset][task][model_name])
elif model_name == "sourcetarget_gae":
    model = methods.get_sourcetarget_gae(**methods.models_suggested_parameters_sets[dataset][task][model_name])
elif model_name == "sourcetarget_vgae":
    model = methods.get_sourcetarget_vgae(**methods.models_suggested_parameters_sets[dataset][task][model_name])



train_data, val_data, test_data = None, None, None
if dataset == "cora":
    if task == "general":
        train_data, val_data, test_data = methods.load_cora(**methods.data_suggested_parameters_sets[dataset][task], device = device)
    elif task == "biased":
        train_data, val_data, test_data = methods.load_cora_biased(**methods.data_suggested_parameters_sets[dataset][task], device = device)


optimizer = torch.optim.Adam(model.parameters(), lr = lr) 


# Array of test set performances
perfs = []

# Compute train and validation set loss normalization
tot_train_edges = train_data.edge_label.size(0)
tot_pos_train_edges = int(train_data.edge_label.sum())
tot_neg_edges_train = tot_train_edges - tot_pos_train_edges
pos_weight = torch.tensor(tot_neg_edges_train / tot_pos_train_edges) # 5
norm = tot_train_edges / (2 * tot_neg_edges_train) 

train_loss = None
if model_name == "gravity_gae"    or  model_name == "sourcetarget_gae" :
    train_loss = methods.StandardLossWrapper(norm,BCEWithLogitsLoss(pos_weight = pos_weight))
elif model_name == "gravity_vgae" or model_name == "sourcetarget_vgae" :
    train_loss = methods.VGAELossWrapper(norm,BCEWithLogitsLoss(pos_weight = pos_weight))


for i in range(num_runs):
    methods.train(train_data, model,  train_loss , optimizer, device, num_epochs, lrscheduler = lrscheduler, early_stopping = early_stopping, val_data = val_data,  val_loss_fn =  val_loss_fn, patience = 50, retrain_data = train_data, use_sparse_representation = use_sparse_representation,  epoch_print_freq = 10)

    perfs.append(methods.evaluate_link_prediction(model, test_data, None, device = device))

# Print test-set performances
mean_std_dict = methods.summarize_link_prediction_evaluation(perfs)
markdown_table = methods.pretty_print_link_performance_evaluation(mean_std_dict, model_name)
print(markdown_table)

### Expected Results

## cora, general

#  gravity_gae
# |       | Gravity-GAE    |
# |:------|:---------------|
# | AUC   | 0.882 +- 0.004 |
# | F1    | 0.728 +- 0.01  |
# | hitsk | 0.741 +- 0.006 |
# | AP    | 0.913 +- 0.003 |

# gravity_vgae
# |       | Gravity-VGAE   |
# |:------|:---------------|
# | AUC   | 0.901 +- 0.004 |
# | F1    | 0.742 +- 0.01  |
# | hitsk | 0.75 +- 0.02   |
# | AP    | 0.912 +- 0.007 |


# sourcetarget_gae
# |       | SourceTarget-GAE   |
# |:------|:-------------------|
# | AUC   | 0.872 +- 0.006     |
# | F1    | 0.48 +- 0.08       |
# | hitsk | 0.62 +- 0.02       |
# | AP    | 0.891 +- 0.005     |


# sourcetarget_vgae
# |       | SourceTarget-VGAE   |
# |:------|:--------------------|
# | AUC   | 0.88 +- 0.01        |
# | F1    | 0.45 +- 0.02        |
# | hitsk | 0.64 +- 0.02        |
# | AP    | 0.899 +- 0.007      |


## cora, biased

# |       | Gravity-GAE    |
# |:------|:---------------|
# | AUC   | 0.82 +- 0.004  |
# | F1    | 0.725 +- 0.005 |
# | hitsk | 0.45 +- 0.01   |
# | AP    | 0.826 +- 0.002 |