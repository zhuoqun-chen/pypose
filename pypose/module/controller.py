import torch
from torch import nn

class Controller(nn.Module):
    def __init__(self):
        pass

    def get_control(self, parameters, state, ref_state, feed_forward_quantity):
        pass
    
    def reset(self):
        pass
