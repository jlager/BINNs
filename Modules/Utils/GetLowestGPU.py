import subprocess, torch, pdb
import numpy as np

def GetLowestGPU(pick_from=[0,1,2,3], verbose=True, return_usages=False):
    
    '''
    Runs nvidia-smi command to pick GPU with lowest memory usage.
    
    Args: 
        pick_from:     integer list of gpus to choose from
        verbose:       boolean for whether to print which device was chosen
        return_usages: boolean for whether to return all GPU memory usages
        
    Returns:
        device: device string (e.g. 'cuda:0' or 'cpu' if no cuda devices)
        usages: optional list of integer memory usage per GPU
    '''
    
    # first, if no GPUs available or provided, return CPU
    if not torch.cuda.is_available() or len(pick_from)==0:
        if verbose:
            print('Device set to cpu')
        return 'cpu'
    
    # if using GPU, run nvidia-smi command from terminal
    nvidia_smi = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE)
    
    # byte string -> utf8 string -> list with each line of nvidia-smi
    nvidia_smi = nvidia_smi.communicate()[0].decode('utf8').split('\n')
    
    # initialize empty list of GPU memory usages
    usages = []

    # parse each line of nvidia-smi output
    for line in nvidia_smi:

        # check if line contains GPU usage statistics
        str_idx = line.find('MiB / ')
        
        # if so, grab memory amount
        if str_idx != -1:
            usages.append(int(line[str_idx-7:str_idx]))

    # sort GPUs based on usage
    gpus_sorted = np.argsort(usages)
    
    # pick GPU with smallest usage from provided list
    for idx in gpus_sorted:
        if idx in pick_from:
            gpu = idx
            break
    device = 'cuda:' + str(gpu)
    
    if verbose:
        print('Device set to ' + device) 
    if return_usages:
        return device, usages
    else:
        return device
