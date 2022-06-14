from __future__ import absolute_import, division, print_function

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import time
import cv2
import sys
import os
from collections import namedtuple
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
from got10k.trackers import Tracker
from tqdm import tqdm
from . import ops
# from .backbones import AlexNetV1
from .fusionbackbone import FusionBackbone
from .heads import SiamFC
from .losses import BalancedLoss, FocalLoss
from .datasets import Pair
from .transforms import SiamFCTransforms
from torch.utils.tensorboard import SummaryWriter

__all__ = ['TrackerSiamFC']


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, padding=0)
        self.batch_norm = nn.BatchNorm2d(1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.batch_norm(out)
        # out = nn.ReLU(inplace=True)(out)
        return out


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
        self.point_wise = depthwise_separable_conv(1280, 1)

    def forward(self, z, x):
        a_z, b_z, c_z, d_z = self.backbone(z)
        a_x, b_x, c_x, d_x = self.backbone(x)
        crr_a = self.head(a_z, a_x)
        crr_a = F.interpolate(crr_a, (17, 17), mode='bicubic')
        crr_b = self.head(b_z, b_x)
        crr_b = F.interpolate(crr_b, (17, 17), mode='bicubic')
        crr_c = self.head(c_z, c_x)
        crr_c = F.interpolate(crr_c, (17, 17), mode='bicubic')
        crr_d = self.head(d_z, d_x)
        crr_d = F.interpolate(crr_d, (17, 17), mode='bicubic')

        # final = crr_a.add(crr_b).add(crr_c).add(crr_d)
        final = torch.cat((crr_a, crr_b, crr_c, crr_d), dim=1)
        # print(final.shape)
        final = self.point_wise(final)
        # print(final.shape)
        return final


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, state=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamfcNAS', True)
        self.cfg = self.parse_args(**kwargs)

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda' if self.cuda else 'cpu')

        # setup model

        self.net = Net(
            backbone=FusionBackbone(state),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)

        # print(self.net.state_dict())

        # load checkpoint if provided
        self.net = self.net.to(self.device)
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage), strict=True)

        # setup criterion
        self.criterion = BalancedLoss()

        # setup optimizer
        self.optimizer = optim.SGD(
            self.net.parameters(),
            lr=self.cfg.initial_lr,
            weight_decay=self.cfg.weight_decay,
            momentum=self.cfg.momentum)

        # setup lr scheduler
        gamma = np.power(
            self.cfg.ultimate_lr / self.cfg.initial_lr,
            1.0 / self.cfg.epoch_num)
        self.lr_scheduler = ExponentialLR(self.optimizer, gamma)

    def parse_args(self, **kwargs):
        # default parameters
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 127,
            'instance_sz': 255,
            'context': 0.5,
            # inference parameters
            'scale_num': 3,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 5,
            'batch_size': 8,
            'num_workers': 32,
            'initial_lr': 1e-4,
            'ultimate_lr': 1e-7,
            'weight_decay': 5e-4,
            'momentum': 0.9,
            'r_pos': 16,
            'r_neg': 0}

        for key, val in kwargs.items():
            if key in cfg:
                cfg.update({key: val})
        return namedtuple('Config', cfg.keys())(**cfg)

    @torch.no_grad()
    def init(self, img, box):
        # set to evaluation mode
        self.net.eval()

        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # create hanning window
        self.upscale_sz = self.cfg.response_up * self.cfg.response_sz
        self.hann_window = np.outer(
            np.hanning(self.upscale_sz),
            np.hanning(self.upscale_sz))
        self.hann_window /= self.hann_window.sum()

        # search scale factors
        self.scale_factors = self.cfg.scale_step ** np.linspace(
            -(self.cfg.scale_num // 2),
            self.cfg.scale_num // 2, self.cfg.scale_num)

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
                    self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(img, axis=(0, 1))
        z = ops.crop_and_resize(
            img, self.center, self.z_sz,
            out_size=self.cfg.exemplar_sz,
            border_value=self.avg_color)
        # cv2.imshow(' ', z)
        # cv2.waitKey()
        #
        # # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        # self.kernel = self.net.backbone(z)
        self.a_z, self.b_z, self.c_z, self.d_z = self.net.backbone(z)

    @torch.no_grad()
    def head_testing(self, z, x):
        result = torch.zeros(size=(self.cfg.scale_num, x.shape[1], self.cfg.response_sz, self.cfg.response_sz), device=self.device)
        for t in range(self.cfg.scale_num):
            current_frame = copy.deepcopy(x[t])
            result[t] = self.net.head(z, current_frame.unsqueeze(0))
            del current_frame
        return result

    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images0
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]

        x = np.stack(x, axis=0)
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()

        # responses
        # import matplotlib.pyplot as plt
        a_x, b_x, c_x, d_x = self.net.backbone(x)
        crr_a = self.head_testing(self.a_z, a_x)
        # print(crr_a.shape)
        # fig, axarr = plt.subplots(1)
        # alpha_pwe = 1
        # crr_a = alpha_pwe * torch.pow(crr_a, 1)
        # axarr.imshow(crr_a.detach().cpu()[0][0], cmap = 'jet')
        # st = 'b_a'
        # plt.title(st)
        # plt.savefig(r'/home/sadegh/Desktop/heatmaps/{}.jpg'.format(st))
        crr_a = F.interpolate(crr_a, (17, 17), mode='bicubic')
        # fig, axarr = plt.subplots(1)
        # axarr.imshow(crr_a.detach().cpu()[0][0], cmap='jet')
        # st = 'a_a'
        # plt.title(st)
        # plt.savefig(r'/home/sadegh/Desktop/heatmaps/{}.jpg'.format(st))
        crr_b = self.head_testing(self.b_z, b_x)
        # print(crr_b.shape)
        # crr_b = alpha_pwe * torch.pow(crr_b, 2)
        # fig, axarr = plt.subplots(1)
        # axarr.imshow(crr_b.detach().cpu()[0][0], cmap='jet')
        # st = 'b_b'
        # plt.title(st)
        # plt.savefig(r'/home/sadegh/Desktop/heatmaps/{}.jpg'.format(st))
        crr_b = F.interpolate(crr_b, (17, 17), mode='bicubic')
        # fig, axarr = plt.subplots(1)
        # axarr.imshow(crr_b.detach().cpu()[0][0], cmap='jet')
        # st = 'a_b'
        # plt.title(st)
        # plt.savefig(r'/home/sadegh/Desktop/heatmaps/{}.jpg'.format(st))
        crr_c = self.head_testing(self.c_z, c_x)
        # print(crr_c.shape)
        # crr_c = alpha_pwe * torch.pow(crr_c, 2)
        # fig, axarr = plt.subplots(1)
        # axarr.imshow(crr_c.detach().cpu()[0][0], cmap='jet')
        # st = 'b_c'
        # plt.title(st)
        # plt.savefig(r'/home/sadegh/Desktop/heatmaps/{}.jpg'.format(st))
        crr_c = F.interpolate(crr_c, (17, 17), mode='bicubic')
        # fig, axarr = plt.subplots(1)
        # axarr.imshow(crr_c.detach().cpu()[0][0], cmap='jet')
        # st = 'a_c'
        # plt.title(st)
        # plt.savefig(r'/home/sadegh/Desktop/heatmaps/{}.jpg'.format(st))
        crr_d = self.head_testing(self.d_z, d_x)
        # print(crr_d.shape)
        # crr_d = alpha_pwe * torch.pow(crr_d, 2)
        # fig, axarr = plt.subplots(1)
        # axarr.imshow(crr_d.detach().cpu()[0][0], cmap='jet')
        # st = 'b_d'
        # plt.title(st)
        # plt.savefig(r'/home/sadegh/Desktop/heatmaps/{}.jpg'.format(st))
        crr_d = F.interpolate(crr_d, (17, 17), mode='bicubic')
        # axarr.imshow(crr_d.detach().cpu()[0][0], cmap='jet')
        # st = 'a_d'
        # plt.title(st)
        # plt.savefig(r'/home/sadegh/Desktop/heatmaps/{}.jpg'.format(st))
        # responses = crr_a.add(crr_b).add(crr_c).add(crr_d)

        responses = torch.zeros(size=(self.cfg.scale_num, 1, self.cfg.response_sz, self.cfg.response_sz))
        for ten_idx in range(self.cfg.scale_num):
            t_a, t_b, t_c, t_d = copy.deepcopy(crr_a[ten_idx]), copy.deepcopy(crr_b[ten_idx]), copy.deepcopy(crr_c[ten_idx]), copy.deepcopy(crr_d[ten_idx])
            cat_dims = torch.cat([t_a.unsqueeze(0), t_b.unsqueeze(0), t_c.unsqueeze(0), t_d.unsqueeze(0)], dim=1)
            responses[ten_idx] = self.net.point_wise(cat_dims)
            del t_a, t_b, t_c, t_d

        ##################################################################
        responses = responses.squeeze(1).cpu().numpy()

        # upsample responses and penalize scale changes
        responses = np.stack([cv2.resize(
            u, (self.upscale_sz, self.upscale_sz),
            interpolation=cv2.INTER_CUBIC)
            for u in responses])
        responses[:self.cfg.scale_num // 2] *= self.cfg.scale_penalty
        responses[self.cfg.scale_num // 2 + 1:] *= self.cfg.scale_penalty

        # peak scale
        scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

        # peak location
        response = responses[scale_id]

        response -= response.min()
        response /= response.sum() + 1e-16
        response = (1 - self.cfg.window_influence) * response + \
                   self.cfg.window_influence * self.hann_window
        loc = np.unravel_index(response.argmax(), response.shape)
        # axarr.imshow(response, cmap='jet')
        # st = 'responses'
        # # responses = alpha_pwe * torch.pow(responses, 2)
        # plt.title(st)
        # plt.savefig(r'/home/sadegh/Desktop/heatmaps/{}.jpg'.format(st))
        # input('sss')
        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
                           self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
                        self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale = (1 - self.cfg.scale_lr) * 1.0 + \
                self.cfg.scale_lr * self.scale_factors[scale_id]
        self.target_sz *= scale
        self.z_sz *= scale
        self.x_sz *= scale

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        return box

    def track(self, img_files, box, visualize=False):
        frame_num = len(img_files)
        boxes = np.zeros((frame_num, 4))
        boxes[0] = box
        times = np.zeros(frame_num)

        for f, img_file in tqdm(enumerate(img_files)):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if visualize:
                ops.show_image(img, boxes[f, :])

        return boxes, times

    def train_step(self, batch, backward=True):
        # set network mode
        self.net.train(backward)

        # parse batch data
        z = batch[0].to(self.device, non_blocking=self.cuda)
        x = batch[1].to(self.device, non_blocking=self.cuda)

        with torch.set_grad_enabled(backward):
            # inference
            responses = self.net(z, x)

            # calculate loss
            labels = self._create_labels(responses.size())
            loss = self.criterion(responses, labels)

            if backward:
                # back propagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return loss.item()

    @torch.enable_grad()
    def train_over(self, seqs, val_seqs=None,
                   save_dir='pretrained', arch_id=0):
        # set to train mode
        self.net.train()
        logs = SummaryWriter()
        # create save_dir folder
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # setup dataset
        transforms = SiamFCTransforms(
            exemplar_sz=self.cfg.exemplar_sz,
            instance_sz=self.cfg.instance_sz,
            context=self.cfg.context)
        dataset = Pair(
            seqs=seqs,
            transforms=transforms)

        # setup dataloader
        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cuda,
            drop_last=True)
        avg_iter_loss = []
        tot_it = 0
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)
            epoch_loss = []

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)

                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                tot_it += it
                logs.add_scalar("Loss_Iteration/train", loss, tot_it)
                sys.stdout.flush()
                epoch_loss.append(loss)
            epoch_loss = np.asarray(epoch_loss)
            logs.add_scalar("Loss_Epock/train", np.mean(epoch_loss), epoch)
            avg_iter_loss.append(np.mean(epoch_loss))
        avg_iter_loss = np.asarray(avg_iter_loss)
        avg_iter_loss = np.mean(avg_iter_loss)
        print('Arc {} Loss : {}'.format(arch_id, avg_iter_loss))

        # save checkpoint
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        net_path = os.path.join(
            save_dir, 'Architecture_number_%d.pth' % arch_id)
        torch.save(self.net.state_dict(), net_path)
        return avg_iter_loss

    def _create_labels(self, size):
        # skip if same sized labels already created
        if hasattr(self, 'labels') and self.labels.size() == size:
            return self.labels

        def logistic_labels(x, y, r_pos, r_neg):
            dist = np.abs(x) + np.abs(y)  # block distance
            labels = np.where(dist <= r_pos,
                              np.ones_like(x),
                              np.where(dist < r_neg,
                                       np.ones_like(x) * 0.5,
                                       np.zeros_like(x)))
            return labels

        # distances along x- and y-axis
        n, c, h, w = size
        x = np.arange(w) - (w - 1) / 2
        y = np.arange(h) - (h - 1) / 2
        x, y = np.meshgrid(x, y)

        # create logistic labels
        r_pos = self.cfg.r_pos / self.cfg.total_stride
        r_neg = self.cfg.r_neg / self.cfg.total_stride
        labels = logistic_labels(x, y, r_pos, r_neg)

        # repeat to size
        labels = labels.reshape((1, 1, h, w))
        labels = np.tile(labels, (n, c, 1, 1))

        # convert to tensors
        self.labels = torch.from_numpy(labels).to(self.device).float()

        return self.labels
