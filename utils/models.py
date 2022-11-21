import torch
import torch.nn as nn
import pydicom
import numpy as np
import segmentation_models_pytorch as smp
import timm
import cv2
import os

import utils.timm4smp as timm4smp

from utils.timm4smp.models.layers.conv2d_same import Conv2dSame
from utils.conv3d_same.conv3d_same import Conv3dSame

import threading

image_size_seg = (128, 128, 128)
msk_size = image_size_seg[0]
image_size_cls = 224
n_slice_per_c = 15
n_ch = 5

def convert_3d(module):

    module_output = module
    if isinstance(module, torch.nn.BatchNorm2d):
        module_output = torch.nn.BatchNorm3d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
            
    elif isinstance(module, Conv2dSame):
        module_output = Conv3dSame(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0]))

    elif isinstance(module, torch.nn.Conv2d):
        module_output = torch.nn.Conv3d(
            in_channels=module.in_channels,
            out_channels=module.out_channels,
            kernel_size=module.kernel_size[0],
            stride=module.stride[0],
            padding=module.padding[0],
            dilation=module.dilation[0],
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode
        )
        module_output.weight = torch.nn.Parameter(module.weight.unsqueeze(-1).repeat(1,1,1,1,module.kernel_size[0]))

    elif isinstance(module, torch.nn.MaxPool2d):
        module_output = torch.nn.MaxPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            ceil_mode=module.ceil_mode,
        )
    elif isinstance(module, torch.nn.AvgPool2d):
        module_output = torch.nn.AvgPool3d(
            kernel_size=module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
        )

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_3d(child)
        )
    del module

    return module_output


n_blocks=4
class TimmSegModel(nn.Module):
    def __init__(self, backbone, segtype='unet', pretrained=False):
        super(TimmSegModel, self).__init__()

        self.encoder = timm4smp.create_model(
            backbone,
            in_chans=3,
            features_only=True,
            pretrained=pretrained
        )
        g = self.encoder(torch.rand(1, 3, 64, 64))
        encoder_channels = [1] + [_.shape[1] for _ in g]
        decoder_channels = [256, 128, 64, 32, 16]
        if segtype == 'unet':
            self.decoder = smp.unet.decoder.UnetDecoder(  
                encoder_channels=encoder_channels[:n_blocks+1],
                decoder_channels=decoder_channels[:n_blocks],
                n_blocks=n_blocks,
            )
        self.segmentation_head = nn.Conv2d(decoder_channels[n_blocks-1], 7, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x, n_blocks = 4):
        global_features = [0] + self.encoder(x)[:n_blocks]
        seg_features = self.decoder(*global_features)
        seg_features = self.segmentation_head(seg_features)
        return seg_features   
    
class TimmModel(nn.Module):
    def __init__(self, backbone, image_size, pretrained=False, in_chans = 6):
        super(TimmModel, self).__init__()
        self.image_size = image_size
        self.encoder = timm.create_model(
            backbone,
            in_chans=in_chans,
            num_classes=1,
            features_only=False,
            drop_rate=0,
            drop_path_rate=0,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone or 'nfnet' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()

        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=0, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
        )
        self.lstm2 = nn.LSTM(hdim, 256, num_layers=2, dropout=0, bidirectional=True, batch_first=True)
        self.head2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
        )


    def forward(self, x, n_slice_per_c=15, in_chans=6):  # (bs, nc*7, ch, sz, sz)
        bs = x.shape[0]
        x = x.view(bs * n_slice_per_c * 7, in_chans, self.image_size, self.image_size)
        feat = self.encoder(x)
        feat = feat.view(bs, n_slice_per_c * 7, -1)
        feat1, _ = self.lstm(feat)
        feat1 = feat1.contiguous().view(bs * n_slice_per_c * 7, 512)
        feat2, _ = self.lstm2(feat)

        return self.head(feat1), self.head2(feat2[:, 0])
    
    
class Timm1BoneModel(nn.Module):
    def __init__(self, backbone, image_size, pretrained=False, in_chans=6):
        super(Timm1BoneModel, self).__init__()
        self.image_size = image_size

        self.encoder = timm.create_model(
            backbone,
            in_chans=in_chans,
            num_classes=1,
            features_only=False,
            drop_rate=0,
            drop_path_rate=0,
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone or 'nfnet' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()

        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=0, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.Dropout(0),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 1),
        )


    def forward(self, x, n_slice_per_c=15, in_chans=6):  # (bs, nslice, ch, sz, sz)
        bs = x.shape[0]
        x = x.view(bs * n_slice_per_c, in_chans, self.image_size, self.image_size)
        feat = self.encoder(x)
        feat = feat.view(bs, n_slice_per_c, -1)
        feat, _ = self.lstm(feat)
        feat = feat.contiguous().view(bs * n_slice_per_c, -1)
        feat = self.head(feat)
        feat = feat.view(bs, n_slice_per_c).contiguous()

        return feat


def load_bone(msk, cid, t_paths, cropped_images, cid_slices):
    n_scans = len(t_paths)
    bone = []
    try:
        msk_b = msk[cid] > 0.2
        msk_c = msk[cid] > 0.05

        x = np.where(msk_b.sum(1).sum(1) > 0)[0]
        y = np.where(msk_b.sum(0).sum(1) > 0)[0]
        z = np.where(msk_b.sum(0).sum(0) > 0)[0]

        if len(x) == 0 or len(y) == 0 or len(z) == 0:
            x = np.where(msk_c.sum(1).sum(1) > 0)[0]
            y = np.where(msk_c.sum(0).sum(1) > 0)[0]
            z = np.where(msk_c.sum(0).sum(0) > 0)[0]

        x1, x2 = max(0, x[0] - 1), min(msk.shape[1], x[-1] + 1)
        y1, y2 = max(0, y[0] - 1), min(msk.shape[2], y[-1] + 1)
        z1, z2 = max(0, z[0] - 1), min(msk.shape[3], z[-1] + 1)
        zz1, zz2 = int(z1 / msk_size * n_scans), int(z2 / msk_size * n_scans)

        inds = np.linspace(zz1 ,zz2-1 ,n_slice_per_c).astype(int)
        inds_ = np.linspace(z1 ,z2-1 ,n_slice_per_c).astype(int)
        slices = []
        for sid, (ind, ind_) in enumerate(zip(inds, inds_)):
            slices.append(ind)
            msk_this = msk[cid, :, :, ind_]

            images = []
            for i in range(-n_ch//2+1, n_ch//2+1):
                try:
                    dicom = pydicom.read_file(t_paths[ind+i])
                    images.append(dicom.pixel_array)
                except:
                    images.append(np.zeros((512, 512)))

            data = np.stack(images, -1)
            data = data - np.min(data)
            data = data / (np.max(data) + 1e-4)
            data = (data * 255).astype(np.uint8)
            msk_this = msk_this[x1:x2, y1:y2]
            xx1 = int(x1 / msk_size * data.shape[0])
            xx2 = int(x2 / msk_size * data.shape[0])
            yy1 = int(y1 / msk_size * data.shape[1])
            yy2 = int(y2 / msk_size * data.shape[1])
            data = data[xx1:xx2, yy1:yy2]
            data = np.stack([cv2.resize(data[:, :, i], (image_size_cls, image_size_cls), interpolation = cv2.INTER_LINEAR) for i in range(n_ch)], -1)
            msk_this = (msk_this * 255).astype(np.uint8)
            msk_this = cv2.resize(msk_this, (image_size_cls, image_size_cls), interpolation = cv2.INTER_LINEAR)

            data = np.concatenate([data, msk_this[:, :, np.newaxis]], -1)

            bone.append(torch.tensor(data))

        cid_slices[cid] = slices
    except:
        for sid in range(n_slice_per_c):
            bone.append(torch.ones((image_size_cls, image_size_cls, n_ch+1)).int())

    cropped_images[cid] = torch.stack(bone, 0)


def load_cropped_images(msk, image_folder, n_ch=n_ch):

    # t_paths = sorted(glob(os.path.join(image_folder, "*")), key=lambda x: int(x.split('/')[-1].split(".")[0]))
    t_paths = []
    slice_paths = sorted(os.listdir(image_folder), key=lambda x: int(x.split(".")[0]))
    for slice_path in slice_paths:
        t_paths.append(image_folder + '/' + slice_path)

    threads = [None] * 7
    cropped_images = [None] * 7
    cid_slices = [None] * 7

    for cid in range(7):
        threads[cid] = threading.Thread(target=load_bone, args=(msk, cid, t_paths, cropped_images, cid_slices))
        threads[cid].start()
    for cid in range(7):
        threads[cid].join()

    return torch.cat(cropped_images, 0), cid_slices