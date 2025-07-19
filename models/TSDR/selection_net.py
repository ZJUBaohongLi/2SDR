import torch
import torch.nn as nn


class SelectionNet(nn.Module):
    def __init__(self, dim_in, dim_out, layer_num):
        super(SelectionNet, self).__init__()
        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_in - (dim_in - dim_out) * i // layer_num,
                          dim_in - (dim_in - dim_out) * (i + 1) // layer_num),
                nn.BatchNorm1d(dim_in - (dim_in - dim_out) * (i + 1) // layer_num),
                nn.ReLU(),
            ) for i in range(layer_num)
        ])
        self.output_label_layer = nn.Sigmoid()
        self.output_layer = nn.Linear(dim_out, 1)

    def forward(self, x):
        x_rep = x.to(torch.float32)
        for layer in self.mlp:
            x_rep = layer(x_rep)
        label = self.output_layer(x_rep)
        return label

