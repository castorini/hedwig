"""
Utils for serialization
"""
import torch


def save_checkpoint(epoch, arch, state_dict, optimizer_state, eval_metric, filename):
    for k, tensor in state_dict.items():
        state_dict[k] = tensor.cpu()

    state = {
        'epoch': epoch,
        'arch': arch,
        'state_dict': state_dict,
        'optimizer_state': None,   # currently do not save optimizer state
        'eval_metric': eval_metric
    }
    torch.save(state, filename)


def load_checkpoint(filename):
    state = torch.load(filename)
    return state['epoch'], state['arch'], state['state_dict'], state['optimizer_state'], state['eval_metric']
