import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearRegression(torch.nn.Module):
    """
    Linear Regressoin Module, the input features and output 
    features are defaults both 1
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.linear = torch.nn.Linear(784, num_classes)
        
    def forward(self,x):
        x = x.view(x.shape[0], -1)
        out = self.linear(x)
        return out