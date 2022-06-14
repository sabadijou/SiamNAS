import torch
import torch.nn as nn
from .nasfpn import NasNet
from .auto_backbone import AutoBackbone


class FusionBackbone(nn.Module):
    def __init__(self, state):
        super(FusionBackbone, self).__init__()

        self.conv_channels = [96, 256, 384, 384]
        layer1 = self.make_one_layer(self.conv_channels[state[0][0][1]],
                                     self.conv_channels[state[0][0][1]],
                                     kernel_size=2,
                                     padding=1,
                                     batch_norm=True)[0]
        layer2 = self.make_one_layer(self.conv_channels[state[0][1][1]],
                                     self.conv_channels[state[0][1][1]],
                                     kernel_size=2,
                                     padding=1,
                                     batch_norm=True)[0]
        layer3 = self.make_one_layer(self.conv_channels[state[0][2][1]],
                                     self.conv_channels[state[0][2][1]],
                                     kernel_size=2,
                                     padding=1,
                                     batch_norm=True)[0]
        layer4 = self.make_one_layer(self.conv_channels[state[0][3][1]],
                                     self.conv_channels[state[0][3][1]],
                                     kernel_size=2,
                                     padding=1,
                                     batch_norm=True)[0]

        feature_maps_dict = dict({
            'fm1': layer1,
            'fm2': layer2,
            'fm3': layer3,
            'fm4': layer4,
        })
        self.backbone = AutoBackbone(state[0])
        self.fpn = NasNet(feature_maps_dict, state[1])

    def forward(self, x):
        x1, x2, x3, x4 = self.backbone(x)
        out = self.fpn(x1, x2, x3, x4)
        return out

    def make_one_layer(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, batch_norm=False):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        if batch_norm:
            layers = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        else:
            layers = [conv2d, nn.ReLU(inplace=True)]
        return layers


# import torch
#
# nas_mat = torch.zeros(size=(4, 4, 2), dtype=torch.int64, device='cuda')
# nas_mat[:, :, 0] = torch.randint(low=0, high=2, size=(4, 4))
# nas_mat[:, :, 1] = torch.randint(low=0, high=4, size=(4, 4))
# nas_mat0 = torch.zeros(size=(4, 4), dtype=torch.int64, device='cuda')
# nas_mat0[:, 0] = torch.randint(low=0, high=2, size=(4,))
# nas_mat0[:, 1] = torch.randint(low=0, high=2, size=(4,))
# nas_mat0[:, 2] = torch.randint(low=0, high=2, size=(4,))
# nas_mat0[:, 3] = torch.randint(low=0, high=2, size=(4,))
# nas_mat0[:, 0] = torch.zeros(4,)
# nas_mat0[:, 1] = torch.zeros(4,)
# nas_mat0[:, 2] = torch.zeros(4,)
# nas_mat0[:, 3] = torch.zeros(4,)
# state = [nas_mat0, nas_mat]
# model = FusionBackbone(state)
# model.to('cuda')
# image = torch.randn(3, 3, 255, 255).to('cuda')
#
# pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(model.parameters)
# # print(pytorch_total_params)
# x = model(image)
# print(x.shape)
