from preprocessing import *
from input_data import *
import torch
from torch import tensor as tt
from torch_geometric import transforms as T
from torch_geometric.data import Data


def load_cora_general(test_percent, val_percent, use_sparse_representation, device):
    # Load data (use same functions as tensorflow's implementation)
    adj_init, features = load_data("cora", "./data/cora.cites" )

    adj, val_edges, val_edges_false, test_edges, test_edges_false =  mask_test_edges_general_link_prediction(adj_init, test_percent, val_percent)

    # Convert scipy matrices to torch_geometric's Data
    features = torch.tensor(features.todense(), dtype = torch.float32)
    train_dense_adjm = torch.tensor((adj + sp.eye(adj.shape[0])).todense())
    edge_label_train_general = train_dense_adjm.reshape(-1)
    train_edge_index = torch.tensor(sparse_to_tuple(adj + sp.eye(adj.shape[0]))[0], dtype = torch.int64).t()

    train_data = Data(x = features, edge_index = train_edge_index, edge_label = edge_label_train_general, edge_label_index = "general")

    val_data = Data( x = features, edge_index = torch.tensor(sparse_to_tuple(adj + sp.eye(adj.shape[0]))[0], dtype = torch.int64).t(), edge_label_index = torch.cat((tt(val_edges), tt(val_edges_false)), dim = 0).t(), edge_label = torch.cat((torch.ones(val_edges.shape[0]), torch.zeros(val_edges_false.shape[0]))))

    test_data = Data( x = features, edge_index = torch.tensor(sparse_to_tuple(adj + sp.eye(adj.shape[0]))[0], dtype = torch.int64).t(), edge_label_index = torch.cat((tt(test_edges), tt(test_edges_false)), dim = 0).t(), edge_label = torch.cat((torch.ones(test_edges.shape[0]), torch.zeros(test_edges_false.shape[0]))))


    # use sparse representation and/or GPU, if required
    if use_sparse_representation:
        tosparse = T.ToSparseTensor()
        train_data = tosparse(train_data)
        test_data = tosparse(test_data).cpu()
        val_data = tosparse(val_data)

    train_data.to(device, "x", "adj_t", "edge_label")
    val_data.to(device,"x", "adj_t","edge_label_index", "edge_label")


    return train_data, val_data, test_data



def load_cora_biased(test_percent, val_percent, use_sparse_representation, rev, device):
    # Load data (use same functions as tensorflow's implementation)
    adj_init, features = load_data("cora", "./data/cora.cites" )
    adj, val_edges, val_edges_false, test_edges, test_edges_false =  mask_test_edges_biased_negative_samples(adj_init, test_percent)


    features = torch.tensor(features.todense(), dtype = torch.float32)
    train_dense_adjm = torch.tensor((adj + sp.eye(adj.shape[0])).todense())
    edge_label_train_general = train_dense_adjm.reshape(-1)
    train_edge_index = torch.tensor(sparse_to_tuple(adj + sp.eye(adj.shape[0]))[0], dtype = torch.int64).t()

    train_data = Data( x = features, edge_index = train_edge_index, edge_label = edge_label_train_general, edge_label_index = "biased")
    if rev:
        train_data.edge_label_index = torch.cat((train_data.edge_index, train_data.edge_index[[1,0],:]), dim = 1)
        num_pos_edges = train_data.edge_index.size(1)
        train_data.edge_label = torch.cat((torch.ones(num_pos_edges), torch.zeros(num_pos_edges)), dim = 0)

    test_data = Data( x = features, edge_index = torch.tensor(sparse_to_tuple(adj + sp.eye(adj.shape[0]))[0], dtype = torch.int64).t(), edge_label_index = torch.cat((tt(test_edges), tt(test_edges_false.copy())), dim = 0).t(), edge_label = torch.cat((torch.ones(test_edges.shape[0]), torch.zeros(test_edges_false.shape[0]))))


    train_data.edge_index =  train_data.edge_index[[1,0], :]
    test_data.edge_index  =  test_data.edge_index[[1,0], :] 


    # use sparse representation and/or GPU, if required
    if use_sparse_representation:
        tosparse = T.ToSparseTensor()
        train_data = tosparse(train_data)
        test_data = tosparse(test_data).cpu()

    train_data.to(device, "x", "adj_t", "edge_label")


    return train_data, None, test_data




def load_cora_bidirectional( use_sparse_representation, device):
    # Load data (use same functions as tensorflow's implementation)
    adj_init, features = load_data("cora", "./data/cora.cites" )

    adj, val_edges, val_edges_false, test_edges, test_edges_false =  mask_test_edges_bidirectional_link_prediction(adj_init)


    features = torch.tensor(features.todense(), dtype = torch.float32)
    train_dense_adjm = torch.tensor((adj + sp.eye(adj.shape[0])).todense())
    edge_label_train_general = train_dense_adjm.reshape(-1)
    train_edge_index = torch.tensor(sparse_to_tuple(adj + sp.eye(adj.shape[0]))[0], dtype = torch.int64).t()

    train_data = Data( x = features, edge_index = train_edge_index, edge_label = edge_label_train_general, edge_label_index = "biased")

    test_data = Data( x = features, edge_index = torch.tensor(sparse_to_tuple(adj + sp.eye(adj.shape[0]))[0], dtype = torch.int64).t(), edge_label_index = torch.cat((tt(test_edges), tt(test_edges_false.copy())), dim = 0).t(), edge_label = torch.cat((torch.ones(test_edges.shape[0]), torch.zeros(test_edges_false.shape[0]))))

    # get the out-degree adjm
    train_data.edge_index =  train_data.edge_index[[1,0], :]
    test_data.edge_index  =  test_data.edge_index[[1,0], :] 



    # use sparse representation and/or GPU, if required
    if use_sparse_representation:
        tosparse = T.ToSparseTensor()
        train_data = tosparse(train_data)
        test_data = tosparse(test_data).cpu()

    train_data.to(device, "x", "adj_t", "edge_label")


    return train_data, None, test_data



data_suggested_parameters_sets = {"cora":{
                                        "general":{"test_percent": 10., "val_percent": 5., "use_sparse_representation": True},
                                        "biased":{"test_percent": 10., "val_percent": None,  "use_sparse_representation": True,"rev":False},
                                        "biased_rev":{"test_percent": 10., "val_percent": None,  "use_sparse_representation": True, "rev":True},
                                        "bidirectional":{ "use_sparse_representation": True},

                                        }
                                }