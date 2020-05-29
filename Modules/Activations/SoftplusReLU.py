import torch, pdb
import torch.nn as nn

class SoftplusReLU(nn.Module):
   
    '''
    Modified Softplus activation function where large values 
    are ReLU activated to prevent floating point blowup.
   
    Args:
        threshold: scalar float for Softplus/ReLU cutoff
   
    Inputs:
        x: torch float tensor of inputs
   
    Returns:
        x: torch float tensor of outputs
    '''
   
    def __init__(self, threshold=20.0):
       
        super().__init__()
        self.threshold = threshold
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
       
    def forward(self, x):
       
        # Softplus for small values, ReLU for large
        x = torch.where(x < 20.0, 
                        self.softplus(x), 
                        self.relu(x))
       
        return x
