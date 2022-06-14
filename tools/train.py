from __future__ import absolute_import

import os
from copy import deepcopy

from got10k.datasets import *

from siamfc import TrackerSiamFC


def finetuning(model, last_model):
    if last_model is not None:
        print('Merging new and old models')
        for new_n, new_p in model.named_parameters():
            for last_n, last_p in last_model.named_parameters():
                if (new_n == last_n) and (new_p.shape == last_p.shape):
                    # print(new_n)
                    new_p.data = last_p.data
    del last_model
    return model


def train_sim(state, last_model, arch_id):
    root_dir = os.path.expanduser(r'/home/sadegh/dataset')
    seqs = GOT10k(root_dir, subset='train', return_meta=False)
    tracker = TrackerSiamFC(state=state)
    # print(tracker.net.parameters)
    num_params = sum(param.numel() for param in tracker.net.parameters())
    print('Number of Model Parameters :', num_params)
    tracker.net = finetuning(tracker.net, last_model.net)
    # tracker.net.load_state_dict(last_model.net.state_dict(), strict=False)
    arc_loss = tracker.train_over(seqs, arch_id=arch_id)
    del seqs
    # del last_model
    return deepcopy(tracker), arc_loss, num_params

# import torch
# backbone_state = torch.LongTensor([[0, 0, 0],
#         [0, 0, 1],
#         [0, 0, 1],
#         [0, 0, 0]])
#     # print(backbone_state.shape)
# fusion_state = torch.LongTensor([[[0, 3],
#          [1, 3],
#          [0, 3],
#          [1, 0]],
#
#         [[1, 2],
#          [1, 2],
#          [1, 2],
#          [0, 0]],
#
#         [[0, 2],
#          [0, 1],
#          [0, 3],
#          [1, 0]],
#
#         [[0, 2],
#          [0, 1],
#          [0, 1],
#          [1, 0]]])
# init_state = [backbone_state, fusion_state]
# train_sim(init_state, '1', 0)