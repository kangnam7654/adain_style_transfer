import os
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).parents[2]
sys.path.append(ROOT_DIR)

import torch.nn as nn
import torch.nn.functional as F
from utils.etc.gram_matrix import gram_matrix


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input