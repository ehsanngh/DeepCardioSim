# Source: https://github.com/neuraloperator/neuraloperator/blob/main/neuralop/data/transforms/base_transforms.py

from abc import abstractmethod

import torch

class Transform(torch.nn.Module):
    """
    Applies transforms or inverse transforms to 
    model inputs or outputs, respectively
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def inverse_transform(self):
        pass

    @abstractmethod
    def cuda(self):
        pass

    @abstractmethod
    def cpu(self):
        pass

    @abstractmethod
    def to(self, device):
        pass