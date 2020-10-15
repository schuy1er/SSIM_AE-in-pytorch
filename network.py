import torch
import torch.nn as nn
from dataset import Dataset
import argparse
from torch.utils.data import DataLoader
import torchvision
import os
import cv2 as cv
from ssim import ssim

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--save', type=str, default='')
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--batchsize', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)

opt = parser.parse_args()

assert torch.cuda.is_available()

if not os.path.exists('./recon_'+opt.save):
    os.mkdir('./recon_'+opt.save)

if not os.path.exists('./model_'+opt.save):
    os.mkdir('./model_'+opt.save)

device = opt.device

def max_ten(mat):
    return torch.max(mat).item()


def cal_ten(mat):
    return torch.sum(mat).item(), torch.sqrt(torch.var(mat)).item()


def min_ten(mat):
    return torch.min(mat).item()


# def compute_ssim(pic_real, pic_fake, ksize=11, k1=0.01, k2=0.03):
#     ssim_loss = 0
#     for i in range(pic_real.shape[0]):
#         for x in range(0, pic_real.shape[2] - ksize + 1):
#             for y in range(0, pic_real.shape[3] - ksize + 1):
#                 ker_real = pic_real[i][0][x:x+ksize][y:y+ksize]
#                 ker_fake = pic_fake[i][0][x:x+ksize][y:y+ksize]
#
#                 ssim_loss += (2*avg_fake*avg_real + k1) * (2*)

class SSIMAE(nn.Module):
    def __init__(self):
        super(SSIMAE, self).__init__()
        self.Encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # output = 64 x 64
            nn.Conv2d(32, 32, 4, 2, 1),  # output = 32 x 32
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.Conv2d(32,64, 4, 2, 1),  # output = 16 x 16
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 128, 4, 2, 1),  # output = 8 x 8
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.Conv2d(32, 100, 8, 1, 0),  # output = 1 x 1
            nn.LeakyReLU(0.2)
        )
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(100, 32, 8, 1, 0),
            nn.ConvTranspose2d(32, 64, 3, 1, 1),
            nn.ConvTranspose2d(64, 128, 3, 1, 1),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ConvTranspose2d(32, 32, 3, 1, 1),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.LeakyReLU(0.2)
        )

    def forward(self, inputs):
        x = self.Encoder(inputs)
        x = self.Decoder(x)
        return x

ssimAE = SSIMAE().to(device)

def train():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor()
    ])
    dataset = Dataset('./data/OK_Orgin_Images/expatch/part1', transform=transform)
    data_loader = DataLoader(dataset=dataset, batch_size=opt.batchsize, shuffle=True, drop_last=True)
    optim = torch.optim.Adam(ssimAE.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    for epoch in range(opt.epoch):
        for batch, data in enumerate(data_loader):
            data = data.to(device)
            optim.zero_grad()
            recon = ssimAE(data)
            loss = ssim(data, recon)
            loss.backward()
            optim.step()
            print('epoch:{}'.format(epoch) + '[{}/{}]'.format(batch, len(dataset) // opt.batchsize))
            print('Loss:{}'.format(loss) + '\n')
        if epoch % 20 == 0:
            data = next(iter(dataset)).to(device)
            data = torch.reshape(data, (-1, 1, 128, 128))
            recon = ssimAE(data)
            torch.save(ssimAE, './model_' + opt.save + '/checkpoint' + '{}'.format(epoch))
            torchvision.utils.save_image(recon, './recon_' + opt.save + '/recon_epoch{}.jpg'.format(epoch),
                                         normalize=True)
            torchvision.utils.save_image(data, './recon_' + opt.save + '/real_epoch{}.jpg'.format(epoch),
                                         normalize=True)


if __name__ == '__main__':
    train()