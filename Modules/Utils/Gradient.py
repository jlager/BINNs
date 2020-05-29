from torch.autograd import grad

def Gradient(outputs, inputs, order=1):

    '''
    Takes the gradient of outputs with respect to inputs up to some order.
    
    Inputs:
        outputs (tensor): function to be differentiated
        inputs  (tensor): differentiation argument
        order      (int): order of the derivative 
        
    Returns:
        grads   (tensor): 
    '''
    
    # return outputs if derivative order is 0
    grads = outputs
    
    # convert outputs to scalar
    outputs = outputs.sum()

    # compute gradients sequentially until order is reached
    for i in range(order):
        grads = grad(outputs, inputs, create_graph=True)[0]
        outputs = grads.sum()

    return grads