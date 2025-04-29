'''
Script with testing functions for pytorch datasets
'''

import torch
import matplotlib.pyplot as plt

from torch.utils.data       import Dataset
from torchvision            import datasets
from torchvision.transforms import ToTensor

# ---------------------------------------------
def test_datasets():
    training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
    )
# ---------------------------------------------
