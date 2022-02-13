from multiprocessing import cpu_count
import torch
from utils.general import LOGGER, colorstr
import torch.nn as nn


def select_device(device, batch_size):
    s = colorstr('yellow','[Device] ')
    
    device = str(device).strip().lower().replace('cuda:','')
    cpu = device == 'cpu'    
    cuda = not cpu and torch.cuda.is_available()
    if cuda:
        devices = device.split(',') if device else '0'  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f'batch-size {batch_size} not multiple of GPU count {n}'
        space = ' ' * (len(s) + 1 - 10)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / 1024 ** 2:.0f}MB)\n"  # bytes to MB
    else:
        s += 'CPU\n'

    LOGGER.info(s) 
    return torch.device('cuda:0' if cuda else 'cpu')
        

def is_parallel(model):
    # Returns True if model is of type DP or DDP
    return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)


def de_parallel(model):
    # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
    return model.module if is_parallel(model) else model