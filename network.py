import torch
import torch.nn as nn
from dataset import Dataset
import argparse
import os
import cv2 as cv

def max_ten(mat):
    return torch.max(mat).item()


def cal_ten(mat):
    return torch.sum(mat).item(), torch.sqrt(torch.var(mat)).item()


def min_ten(mat):
    return torch.min(mat).item()


def compute_ssim(pic_real, pic_fake, ksize=11, k1=0.01, k2=0.03):
    ssim_loss = 0
    for i in range(pic_real.shape[0]):
        for x in range(0, pic_real.shape[2] - ksize + 1):
            for y in range(0, pic_real.shape[3] - ksize + 1):
                ker_real = pic_real[i][0][x:x+ksize][y:y+ksize]
                ker_fake = pic_fake[i][0][x:x+ksize][y:y+ksize]
                
                ssim_loss += (2*avg_fake*avg_real + k1) * (2*)

class SSIMAE(nn.Module):
    def __init__(self):
        super(self).__init__()
        self.Encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),  # output = 64 x 64
            nn.Conv2d(32, 32, 4, 2, 1),  # output = 32 x 32
            nn.Conv2d(32, 32, 4, 1, 0),
            nn.Conv2d(32,64, 4, 2, 1),  # output = 16 x 16
            nn.Conv2d(64, 64, 4, 1, 0),
            nn.Conv2d(64, 128, 4, 2, 1),  # output = 8 x 8
            nn.Conv2d(128, 64, 4, 1, 0),
            nn.Conv2d(64, 32, 4, 1, 0),
            nn.Conv2d(32, 100, 8, 1, 0),  # output = 1 x 1
            nn.LeakyReLU(0.2)
        )
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(100, 32, 8, 1, 0),
            nn.ConvTranspose2d(32, 64, 4, 1, 0),
            nn.ConvTranspose2d(64, 128, 4, 1, 0),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, 64, 4, 1, 0),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ConvTranspose2d(32, 32, 4, 1, 0),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, input):
        x = self.Encoder(input)
        x = self.Decoder(x)