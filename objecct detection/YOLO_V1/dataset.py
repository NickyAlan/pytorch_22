"""
Creates a Pytorch dataset to load the Pascal VOC dataset
"""

import torch
import os
import pandas as pd
from PIL import Image 


class VOCDataset(torch.utils.data.Dataset) :
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform=None) :
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self) :
        return len(self.annotations)

    def __getitem__(self, index) :
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1])
        boxes = []
        with open(label_path) as f :
            for label in f.readlines() :
                class_label, x, y, w, h = [float(x)]