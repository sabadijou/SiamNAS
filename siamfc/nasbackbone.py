import torch

import torch.nn as nn

import torch.nn.functional as F


class NasBackbone(nn.Module):
    def __init__(self, state):
        super(NasBackbone, self).__init__()

        self.blocks_inf = {
            0: {
                'block_id': 0,
                'in_channels': 96,
                'hidden_input': 96,
                'out_channels': 256,
                'kernel_size': 5,
                'stride': 1,
                'groups': 2
            },
            1: {
                'block_id': 1,
                'in_channels': 256,
                'hidden_input': 256,
                'out_channels': 384,
                'kernel_size': 3,
                'stride': 1,
                'groups': 1
            },
            2: {
                'block_id': 2,
                'in_channels': 384,
                'hidden_input': 384,
                'out_channels': 384,
                'kernel_size': 3,
                'stride': 1,
                'groups': 2
            },
            3: {
                'block_id': 3,
                'in_channels': 384,
                'hidden_input': 384,
                'out_channels': 256,
                'kernel_size': 3,
                'stride': 1,
                'groups': 2
            }
        }

        self.input_layer = self.make_one_layer(in_channels=3, out_channels=96, kernel_size=11, stride=2, padding=0,
                                               batch_norm=True)
        # self.input_layer_2 = self.make_one_layer(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1,
        #                                          batch_norm=True)
        # self.input_layer += self.input_layer_2
        self.input_layer.append(nn.MaxPool2d(kernel_size=3, stride=2))
        self.input_layer = nn.Sequential(*self.input_layer)

        self.inp_block_0 = self.make_layers(state[0], self.blocks_inf[0])

        self.inp_block_1 = self.make_layers(state[1], self.blocks_inf[1])

        self.inp_block_2 = self.make_layers(state[2], self.blocks_inf[2])

        self.inp_block_3 = self.make_layers(state[3], self.blocks_inf[3])

    def make_one_layer(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1, batch_norm=False, groups=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups)
        if batch_norm:
            layers = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
        else:
            layers = [conv2d, nn.ReLU(inplace=True)]
        return layers

    def make_layers(self, state, info):
        valid_layers = (state == 1)
        valid_layers = valid_layers.sum()
        if valid_layers == 0:
            layer_list = self.make_one_layer(in_channels=info['in_channels'],
                                             out_channels=info['out_channels'],
                                             kernel_size=info['kernel_size'],
                                             stride=info['stride'],
                                             groups=info['groups'],
                                             batch_norm=True)
        else:
            layer_list = self.make_one_layer(in_channels=info['in_channels'],
                                             out_channels=info['hidden_input'],
                                             kernel_size=info['kernel_size'],
                                             stride=info['stride'],
                                             groups=info['groups'],
                                             batch_norm=True)

        for i in range(0, valid_layers):
            if i != valid_layers - 1:
                layers = self.make_one_layer(in_channels=info['hidden_input'],
                                          out_channels=info['hidden_input'],
                                          kernel_size=info['kernel_size'],
                                          stride=info['stride'],
                                          groups=info['groups'],
                                          padding=1,
                                          batch_norm=True)
            else:

                layers = self.make_one_layer(in_channels=info['hidden_input'],
                                          out_channels=info['out_channels'],
                                          kernel_size=info['kernel_size'],
                                          stride=info['stride'],
                                          groups=info['groups'],
                                          padding=1,
                                          batch_norm=True)
            # print(*layers)
            for mmm in layers:
                layer_list.append(mmm)

        if info['block_id'] == 0:
            layer_list.append(nn.MaxPool2d(kernel_size=3, stride=2))

        # if info['block_id'] != 3:
        #     layer_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.inp_block_0(x)
        x_1 = x
        x = self.inp_block_1(x)
        x_2 = x
        x = self.inp_block_2(x)
        x_3 = x
        x = self.inp_block_3(x)
        # print(x_1.shape, x_2.shape, x_3.shape, x.shape)
        return x_1, x_2, x_3, x
#
# for i in range(0, 10000):
#     nas_mat = torch.zeros(size=(4, 4), dtype=torch.int64, device='cuda')
#     nas_mat[:, 0] = torch.randint(low=0, high=2, size=(4,))
#     nas_mat[:, 1] = torch.randint(low=0, high=2, size=(4,))
#     nas_mat[:, 2] = torch.randint(low=0, high=2, size=(4,))
#     nas_mat[:, 3] = torch.randint(low=0, high=2, size=(4,))
#
#     print(nas_mat)
#     # print(nas_mat)
#     model = NasBackbone(nas_mat)
#
#     # print(model.parameters)
#
#     img = torch.zeros(3, 3, 225, 225)
#     v = model(img)
#     pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     # print(model.parameters)
#     # print(pytorch_total_params)