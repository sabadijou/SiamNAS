import torch
import torch.nn.functional as F
from torchvision import models
import torch.nn as nn
import math


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=0)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=0)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=0)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=0)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=0)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=0)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=0)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=0)
        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

        self.initialize_weights()

        self.relu = nn.ReLU(inplace=True)

        pre_mod = models.vgg16(pretrained=True)

        self.state_dict()['conv1_1.weight'].data[:] = pre_mod.state_dict()['features.0.weight'].data[:]
        self.state_dict()['conv2_1.weight'].data[:] = pre_mod.state_dict()['features.5.weight'].data[:]
        self.state_dict()['conv3_1.weight'].data[:] = pre_mod.state_dict()['features.10.weight'].data[:]
        self.state_dict()['conv4_1.weight'].data[:] = pre_mod.state_dict()['features.17.weight'].data[:]

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        # Block 1
        x = self.relu(self.conv1_1(x))
        x = self.relu(self.conv1_2(x))
        x = self.maxpool(x)
        # Block 2
        x = self.relu(self.conv2_1(x))
        x = self.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x_1 = x
        # Block 3
        x = self.relu(self.conv3_1(x))
        x = self.relu(self.conv3_2(x))
        x = self.relu(self.conv3_3(x))
        x = self.maxpool(x)
        x_2 = x
        # Block 4
        x = self.relu(self.conv4_1(x))
        x = self.relu(self.conv4_2(x))
        x = self.relu(self.conv4_3(x))
        x_3 = x
        # Block 5
        x = self.relu(self.conv5_1(x))

        return x_1, x_2, x_3, x

# a = VGG16()
# image = torch.zeros(2, 3, 255, 255)
# q = a(image)
#
# print(q[0].shape, q[1].shape, q[2].shape, q[3].shape)