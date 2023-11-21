import torch_sparse
import torch
import copy
from torch.nn import Module, ModuleList, Parameter
from torch.nn.functional import dropout
from torch_geometric.utils import add_remaining_self_loops 


class LayerWrapper(Module):
    def __init__(self, layer, normalization_before_activation = None, activation = None, normalization_after_activation =None, dropout_p = None, _add_remaining_self_loops = False, uses_sparse_representation = False  ):
        super().__init__()
        self.activation = activation
        self.normalization_before_activation = normalization_before_activation
        self.layer = layer
        self.normalization_after_activation = normalization_after_activation
        self.dropout_p = dropout_p
        self._add_remaining_self_loops = _add_remaining_self_loops
        self.uses_sparse_representation = uses_sparse_representation

    def forward(self, batch):

    
        new_batch = copy.copy(batch)
        if self._add_remaining_self_loops and not self.uses_sparse_representation:
            new_batch.edge_index, _  = add_remaining_self_loops(new_batch.edge_index)
        elif self._add_remaining_self_loops and self.uses_sparse_representation:
            new_batch.adj_t = torch_sparse.fill_diag(new_batch.adj_t, 2)

        if not self.uses_sparse_representation:
            new_batch.x = self.layer(x = new_batch.x, edge_index = new_batch.edge_index)
        else:

            new_batch.x = self.layer(x = new_batch.x, edge_index = new_batch.adj_t)

        if self.normalization_before_activation is not None:
            new_batch.x = self.normalization_before_activation(new_batch.x)
        if self.activation is not None:
            new_batch.x = self.activation(new_batch.x)
        if self.normalization_after_activation is not None:
            new_batch.x =  self.normalization_after_activation(new_batch.x)
        if self.dropout_p is not None:
            new_batch.x =  dropout(new_batch.x, p=self.dropout_p, training=self.training)


        return new_batch




class GNN_FB(Module):
    def __init__(self, gnn_layers,  preprocessing_layers = [], postprocessing_layers = []):
        super().__init__()

        self.net = torch.nn.Sequential(*preprocessing_layers, *gnn_layers, *postprocessing_layers )

    
    def forward(self, batch):
        return self.net(batch) 




class LinkPropertyPredictorGravity(Module):
    def __init__(self, l, EPS = 1e-2, CLAMP = None, train_l = True): 
        super().__init__()
        self.l_initialization = l
        self.l = Parameter(torch.tensor([l]), requires_grad = train_l )
        self.EPS = EPS
        self.CLAMP = CLAMP
    def forward(self, batch):

        new_batch   = copy.copy(batch)

        if batch.edge_label_index in ["general", "biased", "bidirectional"]: 
            m_i = new_batch.x[:,-1].reshape(-1,1).expand((-1,new_batch.x.size(0))).t()
            r = new_batch.x[:,:-1]

            norm = (r * r).sum(dim = 1, keepdim = True)
            r1r2 = torch.matmul(r, r.t())

            r2 = norm - 2*r1r2 + norm.t() 

            logr2 = torch.log(r2 + self.EPS)

            if self.CLAMP is not None:
                logr2 = logr2.clamp(min = -self.CLAMP, max = self.CLAMP)
            
            new_batch.x = (m_i -  self.l * logr2).reshape(-1)

        else:

            m_j = new_batch.x[new_batch.edge_label_index[1,:],-1]

            diff = new_batch.x[new_batch.edge_label_index[0,:], :-1] - new_batch.x[new_batch.edge_label_index[1,:], :-1]

            r2 = (diff * diff).sum(dim = 1) 
            new_batch.x = m_j - self.l * torch.log(r2 + self.EPS)
            
        return new_batch

    def reset_parameters(self):
        self.l.data = torch.tensor([self.l_initialization]).to(self.l.data.device)




class LinkPropertyPredictorSourceTarget(Module):
    def __init__(self):
        super().__init__()
    def forward(self, batch):

        new_batch   = copy.copy(batch)

        hidden_dimension = batch.x.size(1)
        half_dimension = int(hidden_dimension/2)

        if batch.edge_label_index in ["general", "biased", "bidirectional"] and self.training:

            source = batch.x[:, :half_dimension]
            target = batch.x[:, half_dimension:]

            new_batch.x = torch.matmul(source, target.t()).reshape(-1)

        else:

            new_batch.x = (new_batch.x[new_batch.edge_label_index[0,:], :half_dimension] * new_batch.x[new_batch.edge_label_index[1,:], half_dimension:]).sum(dim = 1).reshape(-1)
            
        return new_batch




class ParallelLayerWrapper(Module):
    def __init__(self, layerwrappers):
        super().__init__()
        self.layerwrappers = ModuleList(layerwrappers)

    def forward(self, batch):
        return [layerwrapper(batch) for layerwrapper in self.layerwrappers]


        