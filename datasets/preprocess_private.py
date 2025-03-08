import numpy as np
from matplotlib import pylab as plt
import nibabel as nib
import random
import glob
import os
from PIL import Image
import imageio
import scipy.ndimage
from scipy.ndimage import generate_binary_structure, binary_dilation, label

def process_label_3d(labels, selected_labels=None):
    """
    处理三维标注数组的函数。
    
    参数:
    labels -- 三维numpy数组，包含不同的标注。
    
    返回:
    processed_labels -- 处理后的三维numpy数组。
    """
    # 逐层处理
    for i in range(labels.shape[2]):
        # 二值化操作
        if selected_labels==None:
            binary_labels = (labels[:,:,i] > 0).astype(np.uint8)
        elif len(selected_labels)>1:
            binary_labels = np.isin(labels[:,:,i], selected_labels).astype(np.uint8)
        else:
            binary_labels = (labels[:,:,i] == selected_labels).astype(np.uint8)
        
        # 定义膨胀结构元素
        structure = generate_binary_structure(2, 1)
        
        # 膨胀操作
        dilated_labels = binary_dilation(binary_labels, structure=structure, iterations=1)
        
        # 标记连通域
        labeled_dilated, num_labels = label(dilated_labels)
        
        # 保留大于20的连通域
        for label_id in range(1, num_labels+1):
            if np.sum(labeled_dilated == label_id) <= 20:
                labeled_dilated[labeled_dilated == label_id] = 0
        
        # 更新标签
        labels[:,:,i] = np.where(labeled_dilated != 0, 1, 0)
        
    
    return labels

def normalize(image, mask=None, percentile_lower=0.2, percentile_upper=99.8):

    if mask is None:
        mask = image != image[0, 0, 0]
    cut_off_lower = np.percentile(image[mask != 0].ravel(), percentile_lower)
    cut_off_upper = np.percentile(image[mask != 0].ravel(), percentile_upper)
    res = np.copy(image)
    res[(res < cut_off_lower) & (mask != 0)] = cut_off_lower
    res[(res > cut_off_upper) & (mask != 0)] = cut_off_upper
    res = (res-res.min()) / (res.max()-res.min())  # 0-1

    return res

def resample_to_spacing(image, label, new_spacing=(1, 1, 1)):
    # 获取原始图像的空间分辨率
    original_spacing = image.header.get_zooms()[:3]
    
    # 计算缩放因子
    zoom_factors = np.array(original_spacing) / np.array(new_spacing)
    
    # 重采样图像
    new_image_data = scipy.ndimage.zoom(image.get_fdata(), zoom_factors, order=3)
    new_label_data = scipy.ndimage.zoom(label.get_fdata(), zoom_factors, order=0)
    
    # 创建新的 NIfTI 图像
    new_image = nib.Nifti1Image(new_image_data, affine=image.affine)
    new_label = nib.Nifti1Image(new_label_data, affine=label.affine)
    
    return new_image, new_label

def center_crop_or_pad(img, target_size):
    h, w = img.shape[:2]
    target_h, target_w = target_size
    
    # 如果图像大于目标尺寸，则进行中心裁剪
    if h > target_h:
        start_h = h // 2 - target_h // 2
        img = img[start_h:start_h + target_h, :]
    if w > target_w:
        start_w = w // 2 - target_w // 2
        img = img[:, start_w:start_w + target_w]
    
    # 如果图像小于目标尺寸，则进行零填充
    if h < target_h:
        pad_h = (target_h - h) // 2
        img = np.pad(img, ((pad_h, target_h - h - pad_h), (0, 0), (0, 0)), mode='constant')
    if w < target_w:
        pad_w = (target_w - w) // 2
        img = np.pad(img, ((0, 0), (pad_w, target_w - w - pad_w), (0, 0)), mode='constant')
    
    return img

nii_list = sorted(glob.glob('./private/images/*.nii.gz'))
label_list = sorted(glob.glob('./private/labels/*.nii.gz'))

data_len = len(nii_list)
train_len = int(data_len*1)
test_len = data_len-train_len


selected_labels = [3,4] # here use the label of SC and CSF

train_path = '../data/private/train/'
test_path = '../data/private/test/'

os.makedirs(train_path,exist_ok=True)
os.makedirs(test_path,exist_ok=True)

count_train = 0
count_test = 0

for i,(nii_path,label_path) in enumerate(zip(nii_list,label_list)):
    print('preprocessing the',i+1,'th subject')
    crop_size = (200,200)
    target_size = (256, 256)
    
    nii_img = nib.load(nii_path)
    label_img = nib.load(label_path)
    
    nii_img, label_img = resample_to_spacing(nii_img, label_img, new_spacing=(1, 1, nii_img.header.get_zooms()[2]))
    
    nii_data = nii_img.get_fdata()
    label_data = label_img.get_fdata()
    print(label_data.shape)
    mask_data = process_label_3d(label_data.copy(),selected_labels)
    
    nii_data = np.transpose(nii_data,(1, 0, 2))
    label_data = np.transpose(label_data,(1, 0, 2))
    mask_data = np.transpose(mask_data,(1, 0, 2))
    
    [h,w] = nii_data.shape[:2]
    
    label_data = label_data.astype(np.uint8)
    mask_data = mask_data.astype(np.uint8)
    
    nii_data = center_crop_or_pad(nii_data,crop_size )
    label_data = center_crop_or_pad(label_data,crop_size )
    mask_data = center_crop_or_pad(mask_data,crop_size )
    
    zoom_factors = target_size[0]/crop_size[0]
    zoom_array = [zoom_factors,zoom_factors,1]
    nii_data = scipy.ndimage.zoom(nii_data, zoom_array, order=3)
    label_data = scipy.ndimage.zoom(label_data, zoom_array, order=0)
    mask_data = scipy.ndimage.zoom(mask_data, zoom_array, order=0)
    
    nii_data = normalize(nii_data)
    
    tensor = np.stack([nii_data, mask_data, label_data])
    
    # 获取 nii 文件名并替换扩展名
    base_name = os.path.basename(nii_path).replace('.nii.gz', '')
    
    if i < train_len:
        for j in range(tensor.shape[3]):
            print(os.path.join(train_path, f'{base_name}_{j + 1}.npy'))
            np.save(os.path.join(train_path, str(count_train + j + 1) + '.npy'), tensor[:,:,:,j])
            #np.save(os.path.join(train_path, f'{base_name}_{j + 1}.npy'), tensor[:,:,:,j])
        count_train = count_train + tensor.shape[3]
    else:
        for j in range(tensor.shape[3]):
            np.save(os.path.join(test_path, str(count_test + j + 1) + '.npy'), tensor[:,:,:,j])
        count_test = count_test + tensor.shape[3]
    
    
    