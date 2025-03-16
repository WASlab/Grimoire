import torch

def StepLRScheduler(optimizer, step_size, gamma=0.1, last_epoch=-1):
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma, last_epoch)
def EpochLrScheduler(optimizer, epochs, lr_decay=0.1, last_epoch=-1):
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, epochs, lr_decay, last_epoch)