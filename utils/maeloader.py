#dataloader for pre-training
import torch
import torchvision 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision    import datasets 
from torchvision import transforms 
from torchvision.utils import save_image 
import torch.utils.data as data 
import numpy as np
from PIL import ImageEnhance,Image
import random
import os
from scipy.ndimage import binary_erosion

def adjust_brightness_contrast(img, brightness_factor=1.0, contrast_factor=1.0):
    """
    调整图像的亮度和对比度
    :param img: 输入图像，numpy数组
    :param brightness_factor: 亮度调整因子，1.0表示不变，小于1.0表示降低亮度，大于1.0表示增加亮度
    :param contrast_factor: 对比度调整因子，1.0表示不变，小于1.0表示降低对比度，大于1.0表示增加对比度
    :return: 调整后的图像，numpy数组
    """
    # 将图像转换为PIL图像
    img = Image.fromarray((img * 255).astype(np.uint8))

    # 调整亮度
    img = ImageEnhance.Brightness(img).enhance(brightness_factor)

    # 调整对比度
    img = ImageEnhance.Contrast(img).enhance(contrast_factor)

    # 转换回numpy数组并归一化
    img = np.array(img).astype(np.float32) / 255.0
    return img

def add_gaussian_noise(img, mean=0, sigma=0.1):
    """
    给图像添加高斯噪声
    :param img: 输入图像，numpy数组
    :param mean: 高斯噪声的均值
    :param sigma: 高斯噪声的标准差
    :return: 添加噪声后的图像，numpy数组
    """
    noise = np.random.normal(mean, sigma, img.shape)
    img = img + noise
    img = np.clip(img, 0, 1)  # 确保图像值在0-1之间
    return img

def normalize(img):
    """
    将图像的灰度值归一化到0-1之间
    :param img: 输入图像，numpy数组
    :return: 归一化后的图像
    """
    img -= img.min()
    img /= img.max()
    return img

def augment_mr_image(img):
    """
    对MR图像进行数据增强
    :param img: 输入图像，numpy数组
    :return: 增强后的图像
    """
    # 调整亮度和对比度
    brightness_factor = np.random.uniform(0.8, 1.2)
    contrast_factor = np.random.uniform(0.8, 1.2)
    img = adjust_brightness_contrast(img, brightness_factor, contrast_factor)
    
    # 添加高斯噪声
    img = add_gaussian_noise(img, mean=0, sigma=0.02)
    
    # 灰度归一化到0-1之间
    img = normalize(img)
    
    return img


def norm(img):
    img -= img.min(1, keepdim=True)[0]
    img /= img.max(1, keepdim=True)[0]
    return img

def cv_random_flip(img):
    # left right flip
    flip_flag = random.randint(0, 2)
    if flip_flag == 1:
        img = np.flip(img, 0).copy()
    if flip_flag == 2:
        img = np.flip(img, 1).copy()
    return img

def randomRotation(image):
    rotate_time = random.randint(0, 3)
    image = np.rot90(image, rotate_time).copy()
    return image

def colorEnhance(image):
    bright_intensity = random.randint(8, 12) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(8, 12) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image

def randomGaussian(img, mean=0.002, sigma=0.002):

    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for _i in range(len(im)):
            im[_i] += random.gauss(mean, sigma)
        return im

    flag = random.randint(0, 3)
    if flag == 1:
        width, height = img.shape
        img = gaussianNoisy(img[:].flatten(), mean, sigma)
        img = img.reshape([width, height])
    return img


def randomPeper(img):
    flag = random.randint(0, 3)
    if flag == 1:
        noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
        for i in range(noiseNum):
            randX = random.randint(0, img.shape[0] - 1)
            randY = random.randint(0, img.shape[1] - 1)
            if random.randint(0, 1) == 0:
                img[randX, randY] = 0
            else:
                img[randX, randY] = 1
    return img

class MAE_Dataset(data.Dataset):
    def __init__(self,img_size,image_root,modality,augment=False,variance_path=None,error_path=None):

        self.modal_list = ['t1', 't2', 't1ce', 'flair', 'gt']
        self.image_root = image_root
        self.modality = modality
        self.images = [self.image_root + f for f in os.listdir(self.image_root) if f.endswith('.npy')]
        self.images.sort(key=lambda x: int(os.path.basename(x).split(".npy")[0]))
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size)
        ])
        self.gt_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(img_size,Image.NEAREST)
        ])
        self.Len = int(len(self.images))
        self.augment = augment
        print('slice number:',self.__len__())
        if variance_path==None:
            self.variance_path = None
        else:
            self.variance_path = variance_path
            self.variance = [os.path.join(self.variance_path , f) for f in os.listdir(self.variance_path) if f.endswith('.npy')]
            self.variance.sort(key=lambda x: int(os.path.basename(x).split(".npy")[0]))
            
        if error_path==None:
            self.error_path = None
        else:
            self.error_path = error_path
            self.error = [os.path.join(self.error_path , f) for f in os.listdir(self.error_path) if f.endswith('.npy')]
            self.error.sort(key=lambda x: int(os.path.basename(x).split(".npy")[0]))

    def __getitem__(self, index):
        if self.modality == 'all':
            modal = int(index / self.Len)
            subject = int(index % self.Len)
            npy = np.load(self.images[subject])
            img = npy[modal, :, :]
  
        elif self.modality == 'spine':
            npy = np.load(self.images[index])
            #print(npy.shape)
            img = npy[0, :, :]
            label = npy[1, :, :]
            seg = npy[2]
            sc_mask = np.zeros_like(label)
            
            sc_mask[np.logical_or(seg == 3, seg == 6)] = 1
            kernel = np.ones((3, 3), dtype=np.uint8)  # Define a 3x3 kernel
            sc_mask = binary_erosion(sc_mask, structure=kernel).astype(np.int8)

        else:
            modal = self.modal_list.index(self.modality)
            npy = np.load(self.images[index])
            img = npy[modal, :, :]

        if self.augment == True:
            img = augment_mr_image(img)  # 使用增强函数
        img = self.img_transform(img)
        label = self.img_transform(label)
        
        if self.variance_path==None:
            return img, label
        elif self.error_path==None:
            variance = np.load(self.variance[index])
            variance = self.img_transform(variance)
            return img, label, variance
        else:
            variance = np.load(self.variance[index])
            variance = self.img_transform(variance)
            error = np.load(self.error[index])
            error = error*sc_mask
            error = self.img_transform(error)

            return img, label, variance, error


    def __len__(self):
        if self.modality == 'all':
            return int(len(self.images))* 4
        else:
            return int(len(self.images))

def get_maeloader(batchsize, shuffle,modality,pin_memory=True,source_modal='t1', target_modal='t2',
        img_size = 256,img_root='../data/train/',num_workers=16,augment=False,variance_path=None,error_path=None):
    dataset = MAE_Dataset(img_size=img_size,image_root=img_root,augment=augment,modality=modality,variance_path=variance_path,error_path=error_path)
    data_loader = data.DataLoader(dataset=dataset, batch_size=batchsize, shuffle=shuffle,pin_memory=pin_memory,num_workers=num_workers)
    return data_loader
