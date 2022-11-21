import os
import pydicom
import cv2
import numpy as np

image_size_seg = (128, 128, 128)

def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = cv2.resize(data, (image_size_seg[0], image_size_seg[1]), interpolation = cv2.INTER_AREA)
    return data


def load_dicom_line_par(path):

    # t_paths = sorted(glob(os.path.join(path, "*")), key=lambda x: int(x.split('/')[-1].split(".")[0]))

    t_paths = []
    slice_paths = sorted(os.listdir(path), key=lambda x: int(x.split(".")[0]))
    for slice_path in slice_paths:
        t_paths.append(path + '/' + slice_path)

    n_scans = len(t_paths)
#     print(n_scans)
    indices = np.quantile(list(range(n_scans)), np.linspace(0., 1., image_size_seg[2])).round().astype(int)
    t_paths = [t_paths[i] for i in indices]

    images = []
    for filename in t_paths:
        images.append(load_dicom(filename))
    images = np.stack(images, -1)
    
    images = images - np.min(images)
    images = images / (np.max(images) + 1e-4)
    images = (images * 255).astype(np.uint8)

    return images  # image 3D 128*128*128