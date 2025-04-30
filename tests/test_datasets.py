'''
Script with testing functions for pytorch datasets
'''

import torch
import pytest
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from deeplearning     import utilities as ut

# ---------------------------------------------
def test_datasets():
    trn_data   = ut.get_data(train= True)
    labels_map = {
        0: 'T-Shirt',
        1: 'Trouser',
        2: 'Pullover',
        3: 'Dress',
        4: 'Coat',
        5: 'Sandal',
        6: 'Shirt',
        7: 'Sneaker',
        8: 'Bag',
        9: 'Ankle Boot',
    }

    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(trn_data), size=(1,)).item()
        img, label = trn_data[sample_idx]

        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label])
        plt.axis('off')
        plt.imshow(img.squeeze(), cmap='gray')
    plt.show()
# ---------------------------------------------
@pytest.mark.parametrize('is_training', [True, False])
def test_dataloader(is_training : bool):
    '''
    Test for Dataloader class
    '''
    data   = ut.get_data(train=is_training)
    loader = DataLoader(data, batch_size=64, shuffle=True)

    features, labels = next(iter(loader))
    print(f'Feature batch shape: {features.size()}')
    print(f'Labels batch shape: {labels.size()}')
    img   = features[0].squeeze()
    label = labels[0]

    plt.imshow(img, cmap='gray')
    plt.show()
    print(f'Label: {label}')
# ---------------------------------------------

