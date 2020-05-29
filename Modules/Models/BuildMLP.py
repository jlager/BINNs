import torch, pdb
import torch.nn as nn

class BuildMLP(nn.Module):
    
    '''
    Builds a standard multilayer perceptron (MLP) with options.
    
    Args:
        input_features: integer number of input features
        layers:         list of integer layer sizes
        activation:     instantiated activation function
        linear_output:  boolean indicator for linear output
    
    Inputs:
        x: torch float tensor of inputs
    
    Returns:
        y: torch float tensor of outputs
    '''
    
    def __init__(self, 
                 input_features, 
                 layers, 
                 activation=None, 
                 linear_output=True,
                 output_activation=None,
                 use_batchnorm=False,
                 dropout_rate=0.0):
        
        # initialization
        super().__init__()
        self.input_features = input_features
        self.layers = layers
        self.activation = activation if activation is not None else nn.Sigmoid()
        self.linear_output = linear_output
        if output_activation is not None:
            self.output_activation = output_activation
        else:
            self.output_activation = self.activation
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate
        
        # hidden layers
        operations = []
        for i, layer in enumerate(layers[:-1]):
            
            # add linear activation
            operations.append(nn.Linear(
                in_features=self.input_features,
                out_features=layer,
                bias=True))
            self.input_features = layer
            
            # batch norm
            if self.use_batchnorm:
                operations.append(nn.BatchNorm1d(layer))
            
            # add nonlinear activation 
            operations.append(self.activation)
            
            # dropout
            if self.dropout_rate > 0:
                operations.append(nn.Dropout(p=self.dropout_rate))
            
        # output layer
        operations.append(nn.Linear(
                in_features=self.input_features,
                out_features=layers[-1],
                bias=True))
        if not self.linear_output:
            operations.append(self.output_activation)
                
        # convert module list to sequential model
        self.MLP = nn.Sequential(*operations)
        
    def forward(self, x):
        
        # run the model
        y = self.MLP(x)
        
        return y
        
        