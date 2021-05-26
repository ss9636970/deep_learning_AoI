import torch
import torch.nn as nn
import numpy as np
import os
import random
import logging

class ResNet(nn.Module): 
    def __init__(self):
        super().__init__()
        # for input chennel * 512 * 512 tensor
        self.conv0 = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=(7, 7), stride=(2, 2), padding=3),
            nn.ReLU(),                   # (512 - 6) / 2 = 253, 253 + 3 = 256
            nn.BatchNorm2d(8, affine=False)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                   # 256
            nn.BatchNorm2d(8, affine=False)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                   # 256
            nn.BatchNorm2d(8, affine=False)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 8, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                  # 256
            nn.BatchNorm2d(8, affine=False)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                  # 128
            nn.BatchNorm2d(16, affine=False)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                  # 128
            nn.BatchNorm2d(16, affine=False)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                  # 128
            nn.BatchNorm2d(16, affine=False)
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                  # 128
            nn.BatchNorm2d(16, affine=False)
        )
        self.conv8 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                  # 64
            nn.BatchNorm2d(32, affine=False)
        )
        self.conv9 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                  # 64
            nn.BatchNorm2d(32, affine=False)
        )
        self.conv10 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                  # 64
            nn.BatchNorm2d(32, affine=False)
        )
        self.conv11 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                  # 64
            nn.BatchNorm2d(32, affine=False)
        )
        self.conv12 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                  # 64
            nn.BatchNorm2d(32, affine=False)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                  # 64
            nn.BatchNorm2d(32, affine=False)
        )
        self.conv14 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                  # 32
            nn.BatchNorm2d(64, affine=False)
        )
        self.conv15 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                  # 32
            nn.BatchNorm2d(64, affine=False)
        )
        self.conv16 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),                  # 32
            nn.BatchNorm2d(64, affine=False)
        )
        self.lin = nn.Sequential(
            nn.Linear(in_features=64*32*32, out_features=300, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=100, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=6, bias=True),
            nn.Softmax(dim=1)
        )

        self.adamax1 = nn.MaxPool2d((2, 2), stride=(2,2))
        self.adamax2 = nn.MaxPool2d((2, 2), stride=(2,2))
        self.adamax3 = nn.MaxPool2d((2, 2), stride=(2,2))

    def forward(self, x):
        n = x.shape[0]
        c0 = self.conv0(x)         # 8 256
        c1 = self.conv1(c0)        # 8 256
        c2 = self.conv2(c1 + c0)        # 8 256
        c3 = self.conv3(c2 + c1)        # 8 256
        c4 = self.conv4(c3 + c2)        # 16 128
        c3_2 = self.adamax1(torch.cat([c3, c3], dim=1))
        c5 = self.conv5(c4 + c3_2)        # 16 128
        c6 = self.conv6(c5 + c4)        # 16 128
        c7 = self.conv7(c6 + c5)        # 16 128
        c8 = self.conv8(c7 + c6)        # 32 64
        c7_2 = self.adamax2(torch.cat([c7, c7], dim=1))
        c9 = self.conv9(c8 + c7_2)        # 32 64
        c10 = self.conv10(c9 + c8)      # 32 64
        c11 = self.conv11(c10 + c9)     # 32 64
        c12 = self.conv12(c11 + c10)     # 32 64
        c13 = self.conv13(c12 + c11)     # 32 64
        c14 = self.conv14(c13 + c12)     # 64 32
        c13_2 = self.adamax3(torch.cat([c13, c13], dim=1))
        c15 = self.conv15(c14 + c13_2)     # 64 32
        c16 = self.conv16(c15 + c14)     # 64 32
        c16 = c16.view(c16.shape[0], -1)
        y = self.lin(c16)
        return y

def tryf():
    print('00000')