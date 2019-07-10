import torch.nn as nn


class WrappedModel(nn.Module):

    def __init__(self, module):
        super(WrappedModel, self).__init__()
        self.module = module


    def forward(self, x):
        return self.module(x)
