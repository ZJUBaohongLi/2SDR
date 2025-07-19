import torch
import torch.nn as nn


class QFunc(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(QFunc, self).__init__()
        self.output_label_layer = nn.Sigmoid()
        self.output_layer = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x_rep = x.to(torch.float32)
        label = self.output_label_layer(self.output_layer(x_rep))
        return label

