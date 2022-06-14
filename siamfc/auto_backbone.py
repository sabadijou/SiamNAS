import torch.nn as nn


class AutoBackbone(nn.Module):
    def __init__(self, state):
        super(AutoBackbone, self).__init__()
        self.state = state
        self.conv_channels = [96, 256, 384, 384]

        self.enc_block_0 = self.encoder(en_state=self.state[0], block_id=0)
        self.enc_block_1 = self.encoder(en_state=self.state[1], block_id=1)
        self.enc_block_2 = self.encoder(en_state=self.state[2], block_id=2)
        #
        self.dec_block_0 = self.decoder(de_state=self.state[0], block_id=0)
        self.dec_block_1 = self.decoder(de_state=self.state[1], block_id=1)
        self.dec_block_2 = self.decoder(de_state=self.state[2], block_id=2)
        self.out_block_ = self.create_out_block(out_state=state[3])
        self.max_pooling = nn.MaxPool2d(2, 2)
        # self.identity = nn.Identity()

    def forward(self, x):
        x = self.enc_block_0(x)
        x_1 = x
        x = self.dec_block_0(x)
        x = self.max_pooling(x)
        # res1 = self.identity(x)
        x = self.enc_block_1(x)
        x_2 = x
        x = self.dec_block_1(x)
        # x += res1
        x = self.max_pooling(x)
        # res2 = self.identity(x)
        x = self.enc_block_2(x)
        x_3 = x
        x = self.dec_block_2(x)
        # x += res2
        x = self.max_pooling(x)
        x = self.out_block_(x)


        return x_1, x_2, x_3, x

    def encoder(self, en_state, block_id):
        layer_list = []
        if block_id == 0:
            if en_state[3] != 0:
                input_layer = self.make_one_layer(in_channels=3,
                                                  out_channels=self.conv_channels[en_state[2]])
                layer_list.extend(input_layer)
            else:
                input_layer = self.make_one_layer(in_channels=3,
                                                  out_channels=self.conv_channels[en_state[1]])
                layer_list.extend(input_layer)
        elif block_id == 1:
            if en_state[3] != 0:
                input_layer = self.make_one_layer(in_channels=self.conv_channels[en_state[0]],
                                                  out_channels=self.conv_channels[en_state[2]])
                layer_list.extend(input_layer)
            else:
                input_layer = self.make_one_layer(in_channels=self.conv_channels[en_state[0]],
                                                  out_channels=self.conv_channels[en_state[1]])
                layer_list.extend(input_layer)
        elif block_id == 2:
            if en_state[3] != 0:
                input_layer = self.make_one_layer(in_channels=self.conv_channels[self.state[1][0]],
                                                  out_channels=self.conv_channels[en_state[2]])
                layer_list.extend(input_layer)
            else:
                input_layer = self.make_one_layer(in_channels=self.conv_channels[self.state[1][0]],
                                                  out_channels=self.conv_channels[en_state[1]])
                layer_list.extend(input_layer)

        for i in range(en_state[3]):
            if i == en_state[3] - 1:
                layers = self.make_one_layer(in_channels=self.conv_channels[en_state[2]],
                                             out_channels=self.conv_channels[en_state[1]])
                layer_list.extend(layers)
            else:
                layers = self.make_one_layer(in_channels=self.conv_channels[en_state[2]],
                                             out_channels=self.conv_channels[en_state[2]])
                layer_list.extend(layers)
        return nn.Sequential(*layer_list)

    def decoder(self, de_state, block_id):
        layer_list = []
        if block_id == 0:
            if de_state[3] != 0:
                input_layer = self.make_one_layer_convtranspose(in_channels=self.conv_channels[self.state[0][1]],
                                                                out_channels=self.conv_channels[de_state[2]])
                layer_list.extend(input_layer)
            else:
                input_layer = self.make_one_layer_convtranspose(in_channels=self.conv_channels[self.state[0][1]],
                                                                out_channels=self.conv_channels[self.state[1][0]])
                layer_list.extend(input_layer)
        elif block_id == 1:
            if de_state[3] != 0:
                input_layer = self.make_one_layer_convtranspose(in_channels=self.conv_channels[self.state[1][1]],
                                                                out_channels=self.conv_channels[de_state[2]])
                layer_list.extend(input_layer)
            else:
                input_layer = self.make_one_layer_convtranspose(in_channels=self.conv_channels[self.state[1][1]],
                                                                out_channels=self.conv_channels[de_state[0]])
                layer_list.extend(input_layer)
        elif block_id == 2:
            if de_state[3] != 0:
                input_layer = self.make_one_layer_convtranspose(in_channels=self.conv_channels[self.state[2][1]],
                                                                out_channels=self.conv_channels[de_state[2]])
                layer_list.extend(input_layer)
            else:
                input_layer = self.make_one_layer_convtranspose(in_channels=self.conv_channels[self.state[2][1]],
                                                                out_channels=256)
                layer_list.extend(input_layer)
        if block_id != 2:
            for i in range(de_state[3]):
                if i == de_state[3] - 1:
                    layers = self.make_one_layer_convtranspose(in_channels=self.conv_channels[de_state[2]],
                                                               out_channels=self.conv_channels[self.state[1][0]])
                    layer_list.extend(layers)
                else:
                    layers = self.make_one_layer_convtranspose(in_channels=self.conv_channels[de_state[2]],
                                                               out_channels=self.conv_channels[de_state[2]])
                    layer_list.extend(layers)
        else:
            for i in range(de_state[3]):
                if i == de_state[3] - 1:
                    layers = self.make_one_layer_convtranspose(in_channels=self.conv_channels[de_state[2]],
                                                               out_channels=256)
                    layer_list.extend(layers)
                else:
                    layers = self.make_one_layer_convtranspose(in_channels=self.conv_channels[de_state[2]],
                                                               out_channels=self.conv_channels[de_state[2]])
                    layer_list.extend(layers)
        return nn.Sequential(*layer_list)

    def make_one_layer(self, in_channels, out_channels, kernel_size=3, padding=0, stride=2, batch_norm=True, groups=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=groups)
        if batch_norm:
            layers = [conv2d,  nn.Dropout(0.5), nn.ReLU(inplace=True), nn.BatchNorm2d(out_channels)]
        else:
            layers = [conv2d, nn.Dropout(0.5), nn.ReLU(inplace=True)]
        return layers

    def make_one_layer_convtranspose(self, in_channels, out_channels, kernel_size=3, padding=0, stride=2, batch_norm=True, groups=1):
        deconv = nn.ConvTranspose2d(in_channels, out_channels,
                                    kernel_size=(kernel_size, kernel_size),
                                    padding=(padding, padding),
                                    stride=(stride, stride),
                                    groups=groups)
        if batch_norm:
            layers = [deconv,  nn.Dropout(0.5), nn.ReLU(inplace=True), nn.BatchNorm2d(out_channels)]
        else:
            layers = [deconv, nn.Dropout(0.5), nn.ReLU(inplace=True)]
        return layers

    def create_out_block(self, out_state):
        if out_state[3] == 0:
           layers = self.make_one_layer(in_channels=256,
                                        out_channels=self.conv_channels[out_state[1]],
                                        kernel_size=7,
                                        stride=1)
        elif out_state[3] == 1:
            layers = self.make_one_layer(in_channels=256,
                                         out_channels=self.conv_channels[out_state[2]],
                                         stride=1,
                                         kernel_size=5)
            layers.extend(self.make_one_layer(in_channels=self.conv_channels[out_state[2]],
                                              out_channels=self.conv_channels[out_state[1]],
                                              stride=1,
                                              kernel_size=3))
        elif out_state[3] == 2:
            layers = self.make_one_layer(in_channels=256,
                                         out_channels=self.conv_channels[out_state[2]],
                                         stride=1,
                                         kernel_size=3)
            layers.extend(self.make_one_layer(in_channels=self.conv_channels[out_state[2]],
                                              out_channels=self.conv_channels[out_state[2]],
                                              stride=1,
                                              kernel_size=3))
            layers.extend(self.make_one_layer(in_channels=self.conv_channels[out_state[2]],
                                              out_channels=self.conv_channels[out_state[1]],
                                              stride=1,
                                              kernel_size=3))
        return nn.Sequential(*layers)


# image = torch.ones(2, 3, 127, 127).to('cuda')
# for i in range(10000):
#     state = [[random.randint(0, 3), random.randint(0, 3), random.randint(0, 3), random.randint(0, 2)],
#              [random.randint(0, 3), random.randint(0, 3), random.randint(0, 3), random.randint(0, 2)],
#              [random.randint(0, 3), random.randint(0, 3), random.randint(0, 3), random.randint(0, 2)],
#              [random.randint(0, 3), random.randint(0, 3), random.randint(0, 3), 2]]
#
#     model = AutoBackbone(state).to('cuda')
#     num_params = sum(param.numel() for param in model.parameters())
#     # print(model.parameters)
#     out = model(image)
#     del model
#     print(True, num_params)
#     if num_params > 1000000:
#         print('big')
# print(model.parameters)