import torch.nn.functional as F
import torch.nn as nn
import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def make_one_layer(in_channels, out_channels, kernel_size=3, padding=0, stride=1, batch_norm=True):
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
    if batch_norm:
        layers = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    else:
        layers = [conv2d, nn.ReLU(inplace=True)]
    return layers


class HeadCreator(nn.Module):
    def __init__(self, model_modules, head_id):
        super(HeadCreator, self).__init__()

        self.head_id = head_id

        idx = 0
        for module in model_modules:
            module_name = 'layer_'+str(idx)
            self.add_module(module_name, module)
            idx += 1

    def forward(self, x):
        x[self.head_id] = list(self.children())[0](x[self.head_id])
        for module in list(self.children())[1:]:
            x[self.head_id] = module(x)
        return x[self.head_id]


class ConcatLayer(nn.Module):
    def __init__(self, based_layer, concat_layer, based_id, concat_id, up_sample=True):
        super(ConcatLayer, self).__init__()

        self.based_layer = based_layer

        self.concat_layer = concat_layer

        self.up_sample = up_sample

        self.based_id = based_id

        self.concat_id = concat_id


    def forward(self, x):

        # if concat layer --> x
        # else --> x[id]

        if isinstance(self.based_layer, ConcatLayer):
            out1 = self.based_layer(x)
        else:
            out1 = self.based_layer(x[self.based_id])
        if isinstance(self.concat_layer, ConcatLayer):
            out2 = self.concat_layer(x)
        else:
            out2 = self.concat_layer(x[self.concat_id])

        transformerconv = nn.Conv2d(out2.shape[1],
                                         out1.shape[1],
                                         stride=(1, 1),
                                         kernel_size=(1, 1),
                                         device=device)
        out2 = transformerconv(out2)

        if self.up_sample:
            # scale = int(out1.shape[2] / out2.shape[2])
            # out2 = nn.Upsample(scale_factor=scale,
            #                    mode='nearest')(out2)
            out2 = torch.nn.functional.interpolate(out2, size=(out1.shape[2], out1.shape[3]))
            # Operator #####################################################################
            # out = torch.concat([out1, out2])
            out = out1.add(out2)
            return out
        else:
            # print(out2.shape, out1.shape)
            # scale = int(out2.shape[2] / out1.shape[2])
            # out2 = nn.MaxPool2d(kernel_size=scale)(out2)
            out2 = F.interpolate(out2, size=(out1.shape[2], out1.shape[3]))
            # Operator #####################################################################
            out = out1.add(out2)
            # out = torch.concat([out2, out1])

        return out


class NetworkCreator:
    def __init__(self, feature_maps, nas_matrix):
        super(NetworkCreator, self).__init__()

        self.feature_maps = feature_maps

        self.nas_matrix = nas_matrix

        self.layers_head_1 = [self.feature_maps['fm1']]

        self.layers_head_2 = [self.feature_maps['fm2']]

        self.layers_head_3 = [self.feature_maps['fm3']]

        self.layers_head_4 = [self.feature_maps['fm4']]

        self.creator()

    def head_x_1(self):
        head_creator = HeadCreator(self.layers_head_1, 0)
        return head_creator

    def head_x_2(self):
        head_creator = HeadCreator(self.layers_head_2, 1)
        return head_creator

    def head_x_3(self):
        head_creator = HeadCreator(self.layers_head_3, 2)
        return head_creator

    def head_x_4(self):
        head_creator = HeadCreator(self.layers_head_4, 3)
        return head_creator

    def creator(self):

        for column_id in range(0, self.nas_matrix.shape[1]):
            self.layers_head_1 += self.make_layers(layer_exist=self.nas_matrix[0, column_id, 0],
                                                   concat_layer_id=self.nas_matrix[0, column_id, 1],
                                                   layer_row_id=0)

        for column_id in range(0, self.nas_matrix.shape[1]):
            self.layers_head_2 += self.make_layers(layer_exist=self.nas_matrix[0, column_id, 0],
                                                   concat_layer_id=self.nas_matrix[0, column_id, 1],
                                                   layer_row_id=1)

        for column_id in range(0, self.nas_matrix.shape[1]):
            self.layers_head_3 += self.make_layers(layer_exist=self.nas_matrix[0, column_id, 0],
                                                   concat_layer_id=self.nas_matrix[0, column_id, 1],
                                                   layer_row_id=2)

        for column_id in range(0, self.nas_matrix.shape[1]):
            self.layers_head_4 += self.make_layers(layer_exist=self.nas_matrix[0, column_id, 0],
                                                   concat_layer_id=self.nas_matrix[0, column_id, 1],
                                                   layer_row_id=3)

        return True

    def make_layers(self, layer_exist, concat_layer_id, layer_row_id):

        layers_list = [
            self.layers_head_1,
            self.layers_head_2,
            self.layers_head_3,
            self.layers_head_4,
        ]

        if layer_exist == 0:
            return []
        # if (layer_row_id == 0) and (concat_layer_id == 1):
        #     return [ConcatLayer(layers_list[layer_row_id][-1],
        #                         layers_list[concat_layer_id][-1],
        #                         based_id=layer_row_id,
        #                         concat_id=concat_layer_id,
        #                         up_sample=True)]
        # if (layer_row_id == 0) and (concat_layer_id == 2):
        #     return [ConcatLayer(layers_list[layer_row_id][-1],
        #                         layers_list[concat_layer_id][-1],
        #                         based_id=layer_row_id,
        #                         concat_id=concat_layer_id,
        #                         up_sample=True)]
        # if (layer_row_id == 0) and (concat_layer_id == 3):
        #     return [ConcatLayer(layers_list[layer_row_id][-1],
        #                         layers_list[concat_layer_id][-1],
        #                         based_id=layer_row_id,
        #                         concat_id=concat_layer_id,
        #                         up_sample=False)]
        # if (layer_row_id == 1) and (concat_layer_id == 0):
        #     return [ConcatLayer(layers_list[layer_row_id][-1],
        #                         layers_list[concat_layer_id][-1],
        #                         based_id=layer_row_id,
        #                         concat_id=concat_layer_id,
        #                         up_sample=False)]
        # if (layer_row_id == 1) and (concat_layer_id == 2):
        #     return [ConcatLayer(layers_list[layer_row_id][-1],
        #                         layers_list[concat_layer_id][-1],
        #                         based_id=layer_row_id,
        #                         concat_id=concat_layer_id,
        #                         up_sample=False)]
        # if (layer_row_id == 1) and (concat_layer_id == 3):
        #     return [ConcatLayer(layers_list[layer_row_id][-1],
        #                         layers_list[concat_layer_id][-1],
        #                         based_id=layer_row_id,
        #                         concat_id=concat_layer_id,
        #                         up_sample=False)]
        # if (layer_row_id == 2) and (concat_layer_id == 0):
        #     return [ConcatLayer(layers_list[layer_row_id][-1],
        #                         layers_list[concat_layer_id][-1],
        #                         based_id=layer_row_id,
        #                         concat_id=concat_layer_id,
        #                         up_sample=False)]
        # if (layer_row_id == 2) and (concat_layer_id == 1):
        #     return [ConcatLayer(layers_list[layer_row_id][-1],
        #                         layers_list[concat_layer_id][-1],
        #                         based_id=layer_row_id,
        #                         concat_id=concat_layer_id,
        #                         up_sample=False)]
        # if (layer_row_id == 2) and (concat_layer_id == 3):
        #     return [ConcatLayer(layers_list[layer_row_id][-1],
        #                         layers_list[concat_layer_id][-1],
        #                         based_id=layer_row_id,
        #                         concat_id=concat_layer_id,
        #                         up_sample=False)]
        # if (layer_row_id == 3) and (concat_layer_id == 0):
        #     return [ConcatLayer(layers_list[layer_row_id][-1],
        #                         layers_list[concat_layer_id][-1],
        #                         based_id=layer_row_id,
        #                         concat_id=concat_layer_id,
        #                         up_sample=False)]
        # if (layer_row_id == 3) and (concat_layer_id == 1):
        #     return [ConcatLayer(layers_list[layer_row_id][-1],
        #                         layers_list[concat_layer_id][-1],
        #                         based_id=layer_row_id,
        #                         concat_id=concat_layer_id,
        #                         up_sample=True)]
        # if (layer_row_id == 3) and (concat_layer_id == 2):
        #     return [ConcatLayer(layers_list[layer_row_id][-1],
        #                         layers_list[concat_layer_id][-1],
        #                         based_id=layer_row_id,
        #                         concat_id=concat_layer_id,
        #                         up_sample=True)]
        if layer_row_id > concat_layer_id:
            return [ConcatLayer(layers_list[layer_row_id][-1],
                                layers_list[concat_layer_id][-1],
                                based_id=layer_row_id,
                                concat_id=concat_layer_id,
                                up_sample=False)]

        if layer_row_id < concat_layer_id:
            return [ConcatLayer(layers_list[layer_row_id][-1],
                                layers_list[concat_layer_id][-1],
                                based_id=layer_row_id,
                                concat_id=concat_layer_id,
                                up_sample=True)]

        if layer_row_id == concat_layer_id:
            return []


class NasNet(nn.Module):

    def __init__(self, feature_maps_dict, nas_matrix):
        super(NasNet, self).__init__()

        self.network_creator = NetworkCreator(feature_maps_dict, nas_matrix)

        self.head_x1 = self.network_creator.head_x_1().to(device=device)

        self.head_x2 = self.network_creator.head_x_2().to(device=device)

        self.head_x3 = self.network_creator.head_x_3().to(device=device)

        self.head_x4 = self.network_creator.head_x_4().to(device=device)

        # self.features1 = make_one_layer(128, 256, kernel_size=1, stride=1, padding='same', batch_norm=True)[0]

        # self.features2 = make_one_layer(512, 128, kernel_size=1, stride=1, padding='same', batch_norm=True)[0]

        # self.features3 = make_one_layer(2048, 256, kernel_size=1, stride=1, padding='same', batch_norm=True)[0]

    def forward(self, input_head_1, input_head_2, input_head_3, input_head_4):
        x = [input_head_1,
             input_head_2,
             input_head_3,
             input_head_4]
        # print(input_head_1.shape, input_head_2.shape,
        #       input_head_3.shape, input_head_4.shape)
        out_1 = self.head_x1(x)
        out_2 = self.head_x2(x)
        out_3 = self.head_x3(x)
        out_4 = self.head_x4(x)
        # print(out_1.shape, out_2.shape, out_3.shape, out_4.shape)
        # x = self.features1(out_1)
        # x = torch.cat((out_2, x), 1)
        # # x = self.features2(x)
        # x = out_3.add(x)
        # # x = self.features3(x)
        # x = out_4.add(x)
        return out_1, out_2, out_3, out_4


# layer1 = make_one_layer(128, 128, kernel_size=3, padding=1, batch_norm=True)[0]
# layer2 = make_one_layer(256, 256, kernel_size=3, padding=1, batch_norm=True)[0]
# layer3 = make_one_layer(512, 512, kernel_size=3, padding=1, batch_norm=True)[0]
# layer4 = make_one_layer(512, 512, kernel_size=3, padding=1, batch_norm=True)[0]
# ##################################################
# feature_maps_dict = dict({
#             'fm1': layer1,
#             'fm2': layer2,
#             'fm3': layer3,
#             'fm4': layer4,
#         })
# nas_mat = torch.zeros(size=(4, 4, 2), dtype=torch.int64, device=device)
# nas_mat[:, :, 0] = torch.randint(low=0, high=2, size=(4, 4))
# nas_mat[:, :, 1] = torch.randint(low=0, high=4, size=(4, 4))
# model = NasNet(feature_maps_dict, nas_mat)
#
# from vgg16 import VGG16
#
# bb = VGG16().to(device)
# image = torch.randn(1, 3, 200, 200).to(device)
#
# x1, x2, x3, x4 = bb(image)
#
#
# out_1, out_2, out_3, out_4 = model(x1, x2, x3, x4)
# print(out_1.shape, 'out_1')
# print(out_2.shape, 'out_2')
# print(out_3.shape, 'out_3')
# print(out_4.shape, 'out_4')
