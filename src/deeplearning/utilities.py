'''
This module contains utility functions
'''

from torchvision            import datasets
from torchvision.transforms import ToTensor
from torch.utils.data       import Dataset

# -------------------------------------------
def get_data(train : bool) -> Dataset:
    '''
    Retrieves a Dataset instance for training or testing
    '''
    data     = datasets.FashionMNIST(
    root     ='data',
    train    = train,
    download = True,
    transform= ToTensor()
    )

    return data
# -------------------------------------------
