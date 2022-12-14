from __future__ import absolute_import, division, print_function
from audioop import reverse
from pickletools import uint8

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

from . import ops
from .backbones import AlexNetV1, AlexNetV4, AlexNetV5, QAlexNetV5, QAlexNetV7, MobileNetV1, resnet18
from .heads import SiamFC
from .losses import BalancedLoss
from .datasets import Pair
from .transforms import SiamFCTransforms


__all__ = ['TrackerSiamFC']


def simulate_quant(x, mul_path=None, add_path=None):

    x = x.cpu().numpy()
    if add_path:
        add = np.load(add_path)
        # print('add:', add)
        x = x - add
    if mul_path:
        mul = 0.0002790141152217984
        # mul = np.load(mul_path)
        # print('mul:', mul)
        x = x / mul

    return np.round(x)


# x in NCHW shape
def to_binarray(x, bitwidth, folding=8, reverse_endian=True, reverse_inner=True, outpath='bytearray.bin'):

    # fold channels
    x = x.astype('int32')
    N, C, H, W = x.shape
    channels_per_fold = C // folding
    x = x.reshape((N, folding, channels_per_fold, H, W))

    # change each number to u2 integer
    result = np.zeros(x.shape, dtype=int)
    mask = x < 0
    result[mask] = x[mask] + (1 << bitwidth)
    result[~mask] = x[~mask]

    # divide each number to bytes
    bytes_per_num = bitwidth // 8
    result_bytes = np.zeros(result.shape + (bytes_per_num,), dtype=int)
    print('result_bytes:', result_bytes.shape)
    for byte in range(bytes_per_num):
        result_bytes[..., byte] = (result >> byte*8) & 255

    # reverse bytes
    if reverse_endian:
        result_bytes = np.flip(result_bytes, axis=-1)
    # reverse channels in each fold
    if reverse_inner:
        result_bytes = np.flip(result_bytes, axis=2)

    # result_bytes in [batch, fold, channels_per_fold, h, w, bytes_per_num] shape
    # transpose to [batch, h, w, fold, channel_per_fold, bytes_per_num]
    result_bytes = result_bytes.transpose((0, 3, 4, 1, 2, 5))
    result_bytes = result_bytes.astype(np.uint8).tobytes()
    # result_bytes = bytearray(result_bytes)
    with open(outpath, "wb") as outfile:
        outfile.write(result_bytes)
    print(outpath, "saved")
    #     for n in range(N):
    #         for h in range(H):
    #             for w in range(W):
    #                 for fold in range(folding):
    #                     for channel in range(channels_per_fold):
    #                         for byte in range(bytes_per_num):
    #                             byte = result_bytes[n, fold, channel, h, w, byte]
    #                             # print(byte)
    #                             byte = bytes(byte)
    #                             outfile.write(byte)


class Net(nn.Module):

    def __init__(self, backbone, head):
        super(Net, self).__init__()
        self.backbone = backbone
        self.head = head
    
    def forward(self, z, x):
        z = self.backbone(z)
        x = self.backbone(x)
        return self.head(z, x)


class TrackerSiamFC(Tracker):

    def __init__(self, net_path=None, **kwargs):
        super(TrackerSiamFC, self).__init__('SiamFC', True)
        self.cfg = self.parse_args(**kwargs)
        self.frame = 0

        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = Net(
            backbone=QAlexNetV5(weights_bitwidth=4, activation_bitwidth=4),  #resnet18(used_layers=[1, 2, 3, 4]),
            head=SiamFC(self.cfg.out_scale))
        ops.init_weights(self.net)
        
        # load checkpoint if provided
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

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
        # cfg = {
        #     # basic parameters
        #     'out_scale': 0.001,
        #     'exemplar_sz': 127,
        #     'instance_sz': 255,
        #     'context': 0.5,
        #     # inference parameters
        #     'scale_num': 1,
        #     'scale_step': 1.0375,
        #     'scale_lr': 0.59,
        #     'scale_penalty': 0.9745,
        #     'window_influence': 0.176,
        #     'response_sz': 17,
        #     'response_up': 16,
        #     'total_stride': 8,
        #     # train parameters
        #     'epoch_num': 50,
        #     'batch_size': 8,
        #     'num_workers': 32,
        #     'initial_lr': 1e-2,
        #     'ultimate_lr': 1e-5,
        #     'weight_decay': 5e-4,
        #     'momentum': 0.9,
        #     'r_pos': 16,
        #     'r_neg': 0}
        cfg = {
            # basic parameters
            'out_scale': 0.001,
            'exemplar_sz': 110,
            'instance_sz': 238,
            'context': 0.5,
            # inference parameters
            'scale_num': 1,
            'scale_step': 1.0375,
            'scale_lr': 0.59,
            'scale_penalty': 0.9745,
            'window_influence': 0.176,
            'response_sz': 17,
            'response_up': 16,
            'total_stride': 8,
            # train parameters
            'epoch_num': 50,
            'batch_size': 8,
            'num_workers': 32,
            'initial_lr': 1e-2,
            'ultimate_lr': 1e-5,
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
        template_pad = self.cfg.instance_sz - self.cfg.exemplar_sz
        padded_z_np = np.pad(z, ((0, template_pad), (0, template_pad), (0, 0)))
        # cv2.imwrite('test_inputs/test_input_238x238_{}.ppm'.format(self.frame), cv2.cvtColor(padded_z_np, cv2.COLOR_BGR2RGB))
        # exemplar features
        z = torch.from_numpy(z).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()
        padded_z = F.pad(z, [0, template_pad, 0, template_pad])
        self.kernel = self.net.backbone(z)
        padded_kernel_quant = simulate_quant(self.net.backbone(padded_z),
                                            mul_path='/home/vision/danilowi/siam_tracking/SIAM_2/iprepo/finn_dev_vision_nopreproc/Mul_0_param0.npy',
                                            add_path='/home/vision/danilowi/siam_tracking/SIAM_2/iprepo/finn_dev_vision_nopreproc/Add_0_param0.npy')
        # to_binarray(padded_kernel_quant, bitwidth=24, outpath='test_inputs/crossing_0.bin')
        self.kernel_quant = simulate_quant(self.kernel,
                                            mul_path='/home/vision/danilowi/siam_tracking/SIAM_2/iprepo/finn_dev_vision_nopreproc/Mul_0_param0.npy',
                                            add_path='/home/vision/danilowi/siam_tracking/SIAM_2/iprepo/finn_dev_vision_nopreproc/Add_0_param0.npy')
        

        # FOR FINN PURPOSES (collect data to run tracker in FPGA)
        save_path = "../XOH/data/Crossing/parameters/"

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        np.save(save_path + 'center.npy', self.center)
        np.save(save_path + 'x_sz.npy', self.x_sz)
        np.save(save_path + 'avg_color.npy', self.avg_color)
        np.save(save_path + 'scale_factors.npy', self.scale_factors)
        np.save(save_path + 'kernel.npy', self.kernel.cpu().detach().numpy())
        np.save(save_path + 'upscale_sz.npy', self.upscale_sz)
        np.save(save_path + 'hann_window.npy', self.hann_window)
        np.save(save_path + 'z_sz.npy', self.z_sz)
        np.save(save_path + 'target_sz.npy', self.target_sz)

    
    @torch.no_grad()
    def update(self, img):
        # set to evaluation mode
        self.net.eval()

        # search images
        x = [ops.crop_and_resize(
            img, self.center, self.x_sz * f,
            out_size=self.cfg.instance_sz,
            border_value=self.avg_color) for f in self.scale_factors]
        # print('tobackbone shape:', [xi.shape for xi in x], x[0].max())
        # cv2.imshow('demo', x[0])
        x = np.stack(x, axis=0)

        # print('tha eshape', x.shape, type(x))
        # np.save('test_input.npy', x)
        self.frame += 1
        cv2.imwrite('test_input_238x238_{}.ppm'.format(self.frame), cv2.cvtColor(x[0], cv2.COLOR_BGR2RGB))
        # x[0] = cv2.cvtColor(x[0], cv2.COLOR_BGR2RGB)
        in_img = x[0]
        
        x = torch.from_numpy(x).to(
            self.device).permute(0, 3, 1, 2).float()

        # responses
        # print(x)
        x = self.net.backbone(x)
        out = simulate_quant(x,
                            mul_path='/home/vision/danilowi/siam_tracking/SIAM_2/iprepo/finn_dev_vision_nopreproc/Mul_0_param0.npy',
                            add_path='/home/vision/danilowi/siam_tracking/SIAM_2/iprepo/finn_dev_vision_nopreproc/Add_0_param0.npy')
        # print('out:', np.min(out), np.max(out), np.unique(out).shape)
        # to_binarray(out, bitwidth=24, outpath='test_inputs/crossing_{}.bin'.format(self.frame))
        # print('kernel:', self.kernel.shape)
        # responses = self.net.head(self.kernel, x)
        # print('exemplar:', self.kernel_quant.shape, 'search region:', out.shape)
        responses = self.net.head(torch.tensor(self.kernel_quant), torch.tensor(out))
        responses = responses.squeeze(1).cpu().numpy()
        responses_sc = responses / (256*256)
        maxpos = np.unravel_index(responses[0].argmax(), responses[0].shape)
        # print('maxpos:', maxpos)

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

        # locate target center
        disp_in_response = np.array(loc) - (self.upscale_sz - 1) / 2
        disp_in_instance = disp_in_response * \
            self.cfg.total_stride / self.cfg.response_up
        disp_in_image = disp_in_instance * self.x_sz * \
            self.scale_factors[scale_id] / self.cfg.instance_sz
        self.center += disp_in_image

        # update target size
        scale =  (1 - self.cfg.scale_lr) * 1.0 + \
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

        for f, img_file in enumerate(img_files):
            img = ops.read_image(img_file)

            begin = time.time()
            if f == 0:
                self.init(img, box)
            else:
                boxes[f, :] = self.update(img)
            times[f] = time.time() - begin

            if boxes[f, 0] < 0 or boxes[f, 1] < 0 or boxes[f, 0] > img.shape[1] or boxes[f, 1] > img.shape[0]:
                print('Object lost. Aborting sequence tracking...')
                # for ff in range(f, frame_num):
                #     boxes[ff, :] = np.array([1, 1, 1, 1])
                break

            if visualize:
                ops.show_image(img, boxes[f, :], fig_n=f)
                if cv2.waitKey(0) == ord('q'):
                    break

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
                   save_dir='pretrained'):
        # set to train mode
        self.net.train()

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
        
        # loop over epochs
        for epoch in range(self.cfg.epoch_num):
            # update lr at each epoch
            self.lr_scheduler.step(epoch=epoch)

            # loop over dataloader
            for it, batch in enumerate(dataloader):
                loss = self.train_step(batch, backward=True)
                print('Epoch: {} [{}/{}] Loss: {:.5f}'.format(
                    epoch + 1, it + 1, len(dataloader), loss))
                sys.stdout.flush()
            
            # save checkpoint
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            net_path = os.path.join(
                save_dir, 'siamfc_QAlexNetV7_I8H4O8A4_e%d.pth' % (epoch + 1))
            torch.save(self.net.state_dict(), net_path)
    
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
