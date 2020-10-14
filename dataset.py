from torch.utils.data import Dataset
import os
import cv2 as cv
from PIL import Image

def read_img(orfile):
    img_list = []
    for filename in os.listdir(orfile):
        out = Image.fromarray(cv.imread(orfile + '/' + filename))
        img_list.append(out)
    return img_list

class Dataset(Dataset):
    def __init__(self, img_list, transform = None):
        # self.orfile = orfile
        self.transform = transform
        self.pic = []
        for x in img_list:
            if self.transform:
                x = self.transform(x)
            self.pic.append(x)

    def __len__(self):
        return len(self.pic)

    def __getitem__(self, index):
        # out = cv.GaussianBlur(out, ksize=(25, 25), sigmaX=0)
        return self.pic[index]
