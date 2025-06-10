import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(GAT, self).__init__()

        self.conv1 = GATConv(in_channels, hidden_channels //2 // num_heads, heads=num_heads)
        self.conv2 = GATConv(hidden_channels//2, out_channels, heads=1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        self.activation = nn.Softmax()

        self.Q = nn.Linear(in_channels, in_channels, bias=False)
        self.K = nn.Linear(in_channels, in_channels, bias=False)
        self.V = nn.Linear(in_channels, in_channels, bias=False)

    def forward(self, node_presentation):
        query = self.Q(node_presentation)
        key = self.K(node_presentation)
        value = self.V(node_presentation)

        # Calculate attention weights
        attention_weights = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = self.activation(attention_weights)

        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, value)
        node = attended_values.sum(-1)
        return node


class GraphAttention(MessagePassing):
    def __init__(self, in_channels, out_channels, activation):
        super().__init__(aggr='add')

        self.act = activation
        self.Q = nn.Linear(in_channels, out_channels)
        self.K = nn.Linear(in_channels, out_channels)
        self.V = nn.Linear(in_channels, out_channels)
    
    def forward(self, x, edge_index):
        query = self.Q(x)
        key = self.K(x)
        value = self.V(x)

        alpha = self.propagate(edge_index, x=(query, key), size=(x.size(0), x.size(0)))
        alpha = self.act(alpha) # softmax

        output = self.propagate(edge_index, x=(alpha, value), size=(x.size(0), x.size(0)))

        return output

    def message(self, x_i, x_j):
        output = torch.matmul(x_i, x_j.transpose(0, 1))

        return output

    
