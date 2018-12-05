""" get sentence
    compute reward
    return reward
"""
import torch

def compute_reward(sent):
    """ Only reward sentences that are less than 15 chars in length for now.
        The reward seems flipped because we are doing gradient ascent, not descent
    """
    return torch.tensor([[ (len(sent) + 0.5)]], requires_grad=True) 