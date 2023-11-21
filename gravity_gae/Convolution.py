from torch.nn import Linear
import torch_sparse
from torch_geometric.nn import MessagePassing


# This conv expects self loops to have already been added, and that the rows of the adjm are NOT normalized by the inverse out-degree +1. Such normalization will be done using the "mean" aggregation inherited from MessagePassing The adjm in question should be the transpose of the usual adjm.
class Conv(MessagePassing):

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr = "mean")
        self.W = Linear(in_channels, out_channels, bias = False)

    def message_and_aggregate(self, adj_t, x):
        return torch_sparse.matmul(adj_t, x, reduce=self.aggr)

    def message(self, x_j):
        return x_j

    def forward(self, x, edge_index):
        transformed = self.W(x)

        transformed_aggregated_normalized = self.propagate(edge_index, x = transformed)  

        return transformed_aggregated_normalized

    def reset_parameters(self):
        super().reset_parameters()
        self.W.reset_parameters()

        