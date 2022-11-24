from rest_framework.views import APIView
from rest_framework.response import Response
# from rest_framework import status

from django.core.files.storage import FileSystemStorage
from django.conf import settings

# from spinefracture.serializers import FileSerializer

class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name

# ML Prediction
import os
import cv2
import pydicom
import numpy as np
import segmentation_models_pytorch as smp

import torch

from utils.load_file import load_dicom_line_par

from utils.models import TimmSegModel, Timm1BoneModel, TimmModel, convert_3d, load_cropped_images

from pydicom.pixel_data_handlers.util import apply_voi_lut

from utils.my_yolov6 import my_yolov6
yolov6_model = my_yolov6("ml_models/best_ckpt_yolo.pt","cpu","utils/mydataset.yaml", 512, True)

device = 'cpu'

image_size_cls = 224
n_slice_per_c = 15


# Stage 1: 
models_seg = [] 
kernel_type = 'timm3d_v2s_unet4b_128_128_128_dsv2_flip12_shift333p7_gd1p5_mixup1_lr1e3_20x50ep'
backbone = 'tf_efficientnetv2_s_in21ft1k'
model_dir_seg = 'ml_models/seg-v2s-0911'
n_blocks = 4
for fold in range(5):
    model = TimmSegModel(backbone, pretrained=False)
    model = convert_3d(model)
    model = model.to(device)
    load_model_file = os.path.join(model_dir_seg, f'{kernel_type}_fold{fold}_best.pth')
    sd = torch.load(load_model_file, map_location=torch.device('cpu'))
    if 'model_state_dict' in sd.keys():
        sd = sd['model_state_dict']
    sd = {k[7:] if k.startswith('module.') else k: sd[k] for k in sd.keys()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    models_seg.append(model)  
print(f'Loading success {len(models_seg)} models')

# Stage 2: Classifier Type 1
kernel_type = '0920_1bonev2_effv2s_224_15_6ch_augv2_mixupp5_drl3_rov1p2_bs8_lr23e6_eta23e6_75ep'
model_dir_cls = 'ml_models/rsna-stage2-type1-v2s-224/'
backbone = 'tf_efficientnetv2_s_in21ft1k'
in_chans = 6
models_cls1 = []
for fold in range(5):
    model = Timm1BoneModel(backbone, image_size=224, pretrained=False)
    load_model_file = os.path.join(model_dir_cls, f'{kernel_type}_fold{fold}_best.pth')
    sd = torch.load(load_model_file, map_location=torch.device('cpu'))
    if 'model_state_dict' in sd.keys():
        sd = sd['model_state_dict']
    sd = {k[7:] if k.startswith('module.') else k: sd[k] for k in sd.keys()}
    model.load_state_dict(sd, strict=True)
    model = model.to(device)
    model.eval()
    models_cls1.append(model)
print(f'Loading success {len(models_cls1)} models')

# Stage 2: Classifier Type 2
kernel_type = '0920_2d_lstmv22headv2_convnn_224_15_6ch_8flip_augv2_drl3_rov1p2_rov3p2_bs4_lr6e6_eta6e6_lw151_50ep'
model_dir_cls = 'ml_models/rsna-stage2-type2-convnn-224/'
backbone = 'convnext_nano'
in_chans = 6
models_cls2 = []

for fold in range(5):
    model = TimmModel(backbone, image_size=224, pretrained=False)
    model = model.to(device)
    load_model_file = os.path.join(model_dir_cls, f'{kernel_type}_fold{fold}_best.pth')
    sd = torch.load(load_model_file, map_location=torch.device('cpu'))
    if 'model_state_dict' in sd.keys():
        sd = sd['model_state_dict']
    sd = {k[7:] if k.startswith('module.') else k: sd[k] for k in sd.keys()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    models_cls2.append(model)
print(f'Loading success {len(models_cls2)} models')

class call_model(APIView):

    def get(self, request):
        return Response("Hello World!")

    def post(sefl, request):
        message = []
        fss = CustomFileSystemStorage(location=settings.UPLOADS_ROOT)
        # try: 

        file_olds = os.listdir(settings.UPLOADS_ROOT)
        if len(file_olds) > 0:
            for file_old in file_olds:
                os.remove(os.path.join(settings.UPLOADS_ROOT , file_old))

        files = request.FILES.getlist('file')
        num_slice = len(files)
        for f in files:
            message.append(f.name)
            _dcm = fss.save(f.name, f)

        path_dcm = str(settings.UPLOADS_ROOT)

        outputs1 = []
        outputs2 = []

        image = load_dicom_line_par(path_dcm)
        if image.ndim < 4:
            image = np.expand_dims(image, 0)
        image = image.astype(np.float32).repeat(3, 0)  # to 3ch
        image = image / 255.
        image = np.expand_dims(image, 0)
        img = torch.tensor(image).float()
        print("Ok")
        with torch.no_grad():
            img = img.to(device)
            # SEG
            pred_masks = []
            for model in models_seg:
                pmask = model(img).sigmoid()
                pred_masks.append(pmask)
            pred_masks = torch.stack(pred_masks, 0).mean(0).cpu().numpy()
        
            # Build cls input
            cls_inp = []

            cropped_images, cid_slices = load_cropped_images(pred_masks[0], path_dcm)
            
            cls_inp.append(cropped_images.permute(0, 3, 1, 2).float() / 255.)
            cls_inp = torch.stack(cls_inp, 0).to(device)  # (1, 105, 6, 224, 224)

            pred_cls1, pred_cls2 = [], []
            # CLS 2
            for _, model in enumerate(models_cls2):
                logits, logits2 = model(cls_inp)
                pred_cls1.append(logits.sigmoid().view(-1, 7, n_slice_per_c))
                pred_cls2.append(logits2.sigmoid())

            # CLS 1
            cls_inp = cls_inp.view(7, 15, 6, image_size_cls, image_size_cls).contiguous()
            for _, model in enumerate(models_cls1):
                logits = model(cls_inp)
                pred_cls1.append(logits.sigmoid().view(-1, 7, n_slice_per_c))

            pred_cls1 = torch.stack(pred_cls1, 0).mean(0)
            pred_cls2 = torch.stack(pred_cls2, 0).mean(0)
            outputs1.append(pred_cls1.cpu())
            outputs2.append(pred_cls2.cpu())

        outputs1 = torch.cat(outputs1)
        outputs2 = torch.cat(outputs2)
        PRED1 = (outputs1.mean(-1)).clamp(0.0001, 0.9999)
        PRED2 = (outputs2.view(-1)).clamp(0.0001, 0.9999)
        pred_1 = list(PRED1.numpy())
        pred_2 = list(PRED2.numpy())

        # PRED1 = [0, 0, 0, 1, 0, 1, 1]

        t_paths = []
        slice_paths = sorted(os.listdir(path_dcm), key=lambda x: int(x.split(".")[0]))
        for slice_path in slice_paths:
            t_paths.append(path_dcm + '/' + slice_path)

        url_slices = []
        for ci in range(7):
            dicom = pydicom.dcmread(t_paths[cid_slices[ci][7]])
            img = apply_voi_lut(dicom.pixel_array, dicom)
            img = img - np.min(img)
            img = img/(np.max(img) + 1e-4)
            img = (img*255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            if PRED1[0, ci] > 0.5:
                imag, ndet = yolov6_model.infer(img, conf_thres=0.4, iou_thres=0.45)    
                write_path = path_dcm + f"/{cid_slices[ci][7]}.png"
                cv2.imwrite(write_path, imag)
            else:
                write_path = path_dcm + f"/{cid_slices[ci][7]}.png"
                cv2.imwrite(write_path, img)
            
            url_slices.append(f"/uploads/{cid_slices[ci][7]}.png")

        print("OKOKOK")
        return Response({   
                    'num_slice': num_slice,  
                    'pred1' : pred_1,
                    'pred2' : pred_2, 
                    'url_c1' : url_slices[0], 
                    'url_c2' : url_slices[1],
                    'url_c3' : url_slices[2],
                    'url_c4' : url_slices[3],
                    'url_c5' : url_slices[4],
                    'url_c6' : url_slices[5],
                    'url_c7' : url_slices[6],        
                })


        # except:
        #     return Response({
        #         "code":500,
        #         "message":"Smt bad has been occured!"
        #     })
