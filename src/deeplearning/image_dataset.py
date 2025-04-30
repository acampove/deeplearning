'''
Adapted from:

https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
'''

import os
import pandas as pd
from torchvision.io     import read_image
from torch.utils.data   import Dataset

class ImageDataset(Dataset):
    '''
    Class meant to implement custom functionality for image processing
    '''
    # --------------------------------
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self._img_labels       = pd.read_csv(annotations_file)
        self._img_dir          = img_dir
        self._transform        = transform
        self._target_transform = target_transform
    # --------------------------------
    def __len__(self):
        return len(self._img_labels)
    # --------------------------------
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
