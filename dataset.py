from torch.utils.data import Dataset
import os
import cv2 as cv
from PIL import Image

class Dataset(Dataset):
    def __init__(self, orfile, transform = None):
        self.orfile = orfile
        self.transform = transform
        self.file = []
        for filename in os.listdir(orfile):
            self.file.append(filename)

    def __len__(self):
        return len(self.file)

    def __getitem__(self, index):
        out = cv.imread(self.orfile + '/' + self.file[index])
        # out = cv.GaussianBlur(out, ksize=(25, 25), sigmaX=0)
        out = Image.fromarray(out)
        if self.transform:
            out = self.transform(out)
            return out
