from siamfc.fusionbackbone import FusionBackbone
from got10k.experiments import *
import torch.nn.functional as F
from siamfc.heads import SiamFC
from siamfc import ops
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import os.path
import torch
import glob
import time
import cv2

device = torch.device('cuda:0' if True else 'cpu')

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=3, padding=1, groups=nin)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class Net(nn.Module):

    def __init__(self, backbone, head, point_wise):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
        self.point_wise = point_wise
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


def init(img, box):

    box = np.array([
        box[1] - 1 + (box[3] - 1) / 2,
        box[0] - 1 + (box[2] - 1) / 2,
        box[3], box[2]], dtype=np.float32)
    center, target_sz = box[:2], box[2:]

    upscale_sz = 272
    hann_window = np.outer(
        np.hanning(upscale_sz),
        np.hanning(upscale_sz))
    hann_window /= hann_window.sum()

    # exemplar and search sizes
    context = 0.5 * np.sum(target_sz)
    z_sz = np.sqrt(np.prod(target_sz + context))
    x_sz = z_sz * (255 / 127)
    # exemplar image
    avg_color = np.mean(img, axis=(0, 1))
    z = ops.crop_and_resize(
        img, center, z_sz,
        out_size=127,
        border_value=avg_color)

    # exemplar features
    z = torch.from_numpy(z).to(device).permute(2, 0, 1).unsqueeze(0).float()
    return z, x_sz, avg_color, target_sz, z_sz, center


def update(img, x_sz, center, avg_color):
    # set to evaluation mode
    # search images
    # print('*')
    scale_factors = 1.0375 ** np.linspace(-(3 // 2), 3 // 2, 3)
    # print('**')
    # x = [ops.crop_and_resize(
    #     img, center, x_sz * f,
    #     out_size=255,
    #     border_value=avg_color) for f in scale_factors]
    x = []
    for i in scale_factors:
        eq = ops.crop_and_resize(img=img,
                                 center=center,
                                 size=x_sz * i,
                                 out_size=255,
                                 border_value=avg_color)
        x.append(eq)
    # print('***')
    x = np.stack(x, axis=0)
    # print('****')
    x = torch.from_numpy(x).to(device).permute(0, 3, 1, 2).float()
    # print('*****')
    return x


def generate_box(responses, x_sz, center, target_sz, z_sz):
    # upsample responses and penalize scale changes
    scale_factors = 1.0375 ** np.linspace(-(3 // 2), 3 // 2, 3)
    responses = np.stack([cv2.resize(
        u, (272, 272),
        interpolation=cv2.INTER_CUBIC)
        for u in responses])
    responses[:3 // 2] *= 0.9745
    responses[3 // 2 + 1:] *= 0.9745
    # print('7')
    # peak scale
    scale_id = np.argmax(np.amax(responses, axis=(1, 2)))

    hann_window = np.outer(
        np.hanning(272),
        np.hanning(272))
    hann_window /= hann_window.sum()
    # print('8')
    # peak location
    response = responses[scale_id]
    response -= response.min()
    response /= response.sum() + 1e-16
    response = (1 - 0.176) * response + \
               0.176 * hann_window
    loc = np.unravel_index(response.argmax(), response.shape)
    # print('9')
    # locate target center
    disp_in_response = np.array(loc) - (272 - 1) / 2
    disp_in_instance = disp_in_response * (8 / 16)
    disp_in_image = disp_in_instance * x_sz * scale_factors[scale_id] / 255
    center += disp_in_image
    # print('10')
    # update target size
    scale = (1 - 0.59) * 1.0 + \
            0.59 * scale_factors[scale_id]
    target_sz *= scale
    z_sz *= scale
    x_sz *= scale
    # print('11')
    # return 1-indexed and left-top based bounding box
    box = np.array([
        center[1] + 1 - (target_sz[1] - 1) / 2,
        center[0] + 1 - (target_sz[0] - 1) / 2,
        target_sz[1], target_sz[0]])
    # print('12')
    return box, center


@torch.no_grad()
def arch_tester(state, tracker_id):
    data_path = r'/home/sadegh/dataset/val'
    net_path = r'pretrained/Architecture_number_{}.pth'.format(str(tracker_id))
    net = Net(
        backbone=FusionBackbone(state),
        head=SiamFC(0.001),
        point_wise= depthwise_separable_conv(4, 1)
             )
    net.load_state_dict(torch.load(net_path), strict=True)
    net.eval()
    net.to(device)

    with open(os.path.join(data_path, 'list.txt')) as seqs_add:
        seqs_address = seqs_add.read().split('\n')

    for item in seqs_address:
        print(item)
        # Load GrandTruth Bboxes
        gt_bbox = np.loadtxt(data_path + '/' + item + '/groundtruth.txt', delimiter=',')
        # Load Frames address
        frames = glob.glob(os.path.join(data_path, item, '*.jpg'))
        # Load Kernel
        z, x_sz, avg_color, target_sz, z_sz, center = init(ops.read_image(frames[0]), np.asarray(gt_bbox[0]))
        a_z, b_z, c_z, d_z = net.backbone(z)
        # print('1')
        # Load Frames
        seq_times = np.zeros(shape=(len(frames), ))
        pred_bboxes = np.zeros(shape=(len(frames), 4))
        pred_bboxes[0] = gt_bbox[0]
        counter = 1
        os.makedirs(os.path.join(r'sim_results', 'results', str(tracker_id), 'siam-fusion', item))
        for frame in tqdm(frames[1:]):
            img = ops.read_image(frame)
            # print('2')
            start_time = time.time()
            x = update(img, x_sz, center, avg_color)
            # print('3')
            a_x, b_x, c_x, d_x = net.backbone(x)
            # print('4')
            #####################################################
            # responses = net.head(z, x)
            crr_a = net.head(a_z, a_x)
            crr_a = F.interpolate(crr_a, (17, 17), mode='bicubic')
            crr_b = net.head(b_z, b_x)
            crr_b = F.interpolate(crr_b, (17, 17), mode='bicubic')
            crr_c = net.head(c_z, c_x)
            crr_c = F.interpolate(crr_c, (17, 17), mode='bicubic')
            crr_d = net.head(d_z, d_x)
            crr_d = F.interpolate(crr_d, (17, 17), mode='bicubic')
            # responses = crr_a.add(crr_b).add(crr_c).add(crr_d)
            cat_dims = torch.cat((crr_a, crr_b, crr_c, crr_d), dim=1)
            responses = net.point_wise(cat_dims)
            #####################################################
            # print('5')
            responses = responses.squeeze(1).cpu().numpy()
            # print('6')
            box, center = generate_box(responses, x_sz, center, target_sz, z_sz)
            end_time = time.time()
            seq_times[counter] = end_time - start_time
            pred_bboxes[counter] = box
            counter += 1
        np.savetxt(fname=os.path.join(r'sim_results', 'results', str(tracker_id),
                               'siam-fusion', item, item +'_001.{}'.format('txt')),
                   X=pred_bboxes, delimiter=',', fmt='%.3f')
        np.savetxt(fname=os.path.join(r'sim_results', 'results', str(tracker_id),
                                      'siam-fusion', item, item +'_time.{}'.format('txt')),
                   X=seq_times, delimiter=',', fmt='%.5f')

    e = ExperimentGOT10k(r'/home/sadegh/dataset', subset='val')
    e.result_dir = r'sim_results/results/{}'.format(str(tracker_id))
    e.report_dir = r'sim_results/reports/{}'.format(str(tracker_id))
    results = e.report(['siam-fusion'])
    del net
    del e
    torch.cuda.empty_cache()
    return results['siam-fusion']['overall']['ao'], results['siam-fusion']['overall']['sr'], results['siam-fusion']['overall']['speed_fps']


#
# backbone_state = torch.LongTensor([[0, 0],
#                                    [0, 0],
#                                    [0, 1],
#                                    [1, 1]])
# fusion_state = torch.LongTensor([[[1, 3],
#                                   [1, 3],
#                                   [1, 3]],
#
#                                  [[1, 0],
#                                   [0, 0],
#                                   [1, 2]],
#
#                                  [[1, 3],
#                                   [0, 1],
#                                   [0, 3]],
#
#                                  [[0, 3],
#                                   [0, 1],
#                                   [0, 2]]])
#
# init_state = [backbone_state, fusion_state]
#
# arch_tester(state=init_state, tracker_id=0)
