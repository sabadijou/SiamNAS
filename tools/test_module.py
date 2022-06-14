from __future__ import absolute_import

import os

import torch.cuda
from got10k.experiments import *
from siamfc import TrackerSiamFC

def test_tracker(state, tracker_id):
    path = r'/home/sadegh/dataset'
    path = os.path.expanduser(path)
    # e = ExperimentGOT10k(path, subset='val')
    e = ExperimentOTB(root_dir=r'/home/sadegh/dataset/otb2015', version=2015)
    e.result_dir = r'sim_results/results/{}'.format(str(tracker_id))
    e.report_dir = r'sim_results/reports/{}'.format(str(tracker_id))
    net_path = r'/home/sadegh/PycharmProjects/siam-cuda01/tools/pretrained/Architecture_number_{}.pth'.format(str(tracker_id))
    tracker = TrackerSiamFC(net_path=net_path, state=state)
    e.run(tracker)
    a = e.report([tracker.name])
    del tracker.net
    del tracker
    del e
    torch.cuda.empty_cache()
    print(a['SiamfcNAS']['overall']['success_score'],\
           a['SiamfcNAS']['overall']['precision_score'], a['SiamfcNAS']['overall']['speed_fps'])
    return a['SiamfcNAS']['overall']['success_score'],\
           a['SiamfcNAS']['overall']['precision_score'], a['SiamfcNAS']['overall']['speed_fps']

    # path = os.path.expanduser(path)
    # e = ExperimentGOT10k(path, subset='val')
    # e.result_dir = r'sim_results/results_thesis/{}'.format(str(tracker_id))
    # e.report_dir = r'sim_results/reports_thesis/{}'.format(str(tracker_id))
    # net_path = r'/home/sadegh/PycharmProjects/siam-cuda01/tools/pretrained/Architecture_number_{}.pth'.format(str(tracker_id))
    # tracker = TrackerSiamFC(net_path=net_path, state=state)
    # # print(tracker.net.parameters)
    # e.run(tracker)
    # a = e.report([tracker.name])
    # del tracker.net
    # del tracker
    # del e
    # torch.cuda.empty_cache()
    # print(a['SiamfcNAS']['overall']['ao'], a['SiamfcNAS']['overall']['sr'], a['SiamfcNAS']['overall']['speed_fps'])
    # return a['SiamfcNAS']['overall']['ao'], a['SiamfcNAS']['overall']['sr'], a['SiamfcNAS']['overall']['speed_fps']
    #
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
#
# test_tracker(init_state, 48)