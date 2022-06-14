from __future__ import absolute_import

import os
import glob
import numpy as np

from siamfc import TrackerSiamFC

import torch

if __name__ == '__main__':
    backbone_state = torch.LongTensor([[0, 0, 0],
                                       [0, 0, 1],
                                       [0, 0, 1],
                                       [0, 0, 0]])
    # print(backbone_state.shape)
    fusion_state = torch.LongTensor([[[0, 3],
                                      [1, 3],
                                      [0, 3],
                                      [1, 0]],

                                     [[1, 2],
                                      [1, 2],
                                      [1, 2],
                                      [0, 0]],

                                     [[0, 2],
                                      [0, 1],
                                      [0, 3],
                                      [1, 0]],

                                     [[0, 2],
                                      [0, 1],
                                      [0, 1],
                                      [1, 0]]])
    init_state = [backbone_state, fusion_state]

    state = [backbone_state, fusion_state]
    seq_dir = os.path.expanduser(r'/home/sadegh/data/Basketball/')
    img_files = sorted(glob.glob(seq_dir + 'img/*.jpg'))
    # anno = np.loadtxt(seq_dir + 'groundtruth_rect.txt')
    anno = np.asarray([[198, 214, 34, 81]])

    net_path = r'pretrained/Architecture_number_48.pth'
    tracker = TrackerSiamFC(net_path=net_path, state=state)
    tracker.track(img_files, anno[0], visualize=True)
