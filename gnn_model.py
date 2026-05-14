import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()

        self.conv1 = GCNConv(in_channels=5, out_channels=hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

        self.lin1 = torch.nn.Linear(hidden_dim, 64)
        self.lin2 = torch.nn.Linear(64, 1)

    def forward(self, x, edge_index, batch):

        # --- Graph Convolution ---
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        # --- pooling (graph → vector) ---
        x = global_mean_pool(x, batch)

        # --- MLP head ---
        x = self.lin1(x)
        x = F.relu(x)

        x = self.lin2(x)

        return x