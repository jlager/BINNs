import sys, time
import numpy as np
from datetime import timedelta

def TimeRemaining(current_iter, 
                  total_iter, 
                  start_time, 
                  previous_time=None, 
                  ops_per_iter=1.0):
    
    '''
    Computes time remaining in a loop.
    
    Args:
        current_iter:  integer for current iteration number
        total_iter:    integer for total number of iterations
        start_time:    float initial time
        previous_time: float time of previous iteration
        ops_per_iter:  integer number of operations per iteration
        
    Returns:
        elapsed:   string of elapsed time
        remaining: string of remaining time
        ms_per_op: optional string of milliseconds per operation
    '''
    
    # compute elapsed and remaining time
    current_time = time.time()
    elapsed = current_time - start_time
    remaining = total_iter * elapsed / current_iter - elapsed
    
    # compute optional time between operations
    ms_per_op = None
    if previous_time is not None:
        ms_per_op = (current_time - previous_time) / ops_per_iter
    
    # convert seconds to datetime
    elapsed = str(timedelta(seconds=int(elapsed)))
    remaining = str(timedelta(seconds=int(remaining)))
    
    # convert seconds to milliseconds
    if ms_per_op is not None:
        ms_per_op = '{0}'.format(int(np.round(ms_per_op * 1000)))
        
    return elapsed, remaining, ms_per_op