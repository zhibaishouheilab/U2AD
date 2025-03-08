# 工具函数，专门用来给每次迭代后的模型生成对应的方差图，用于后续的迭代优化

import torch
from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms
import numpy as np
import os
from utils.maeloader import *
from U2AD.model.UGMAE import *
from utils.mae_visualize import *
from options import Pretrain_Private_Options
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal
from scipy.ndimage import generate_binary_structure, binary_dilation, binary_erosion
from scipy.ndimage import label as Label
import torchvision.transforms.functional as F


def load_model(checkpoint_path, device, opt):
    model = EdgeMAE(img_size=opt.img_size, patch_size=opt.patch_size, embed_dim=opt.dim_encoder, depth=opt.depth, 
                    num_heads=opt.num_heads, in_chans=1, decoder_embed_dim=opt.dim_decoder, 
                    decoder_depth=opt.decoder_depth, decoder_num_heads=opt.decoder_num_heads, 
                    mlp_ratio=opt.mlp_ratio, norm_pix_loss=False, patchwise_loss=opt.use_patchwise_loss)
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=False)
    model = model.to(device)
    model.eval()
    return model

def preprocess_npy_image(npy_path, img_size, modality='all'):
    npy = np.load(npy_path)
    modal_list = ['t1', 't2', 't1ce', 'flair', 'gt']
    if modality == 'all':
        img = np.concatenate([npy[modal_list.index(mod), :, :][np.newaxis, :, :] for mod in modal_list[:-1]], axis=0)
    elif modality == 'spine':
        img = npy[0, :, :][np.newaxis, :, :]
        mask = npy[1, :, :][np.newaxis, :, :]
        label = npy[2, :, :][np.newaxis, :, :]
    else:
        img = npy[modal_list.index(modality), :, :][np.newaxis, :, :]
    

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((img_size, img_size))
    ])
    
    img = torch.tensor(img, dtype=torch.float32)
    img = torch.stack([transform(Image.fromarray(img[ch].numpy())) for ch in range(img.shape[0])], dim=0)
    mask = torch.tensor(mask)
    mask = torch.stack([transform(Image.fromarray(mask[ch].numpy())) for ch in range(mask.shape[0])], dim=0)
    label = torch.tensor(label)
    label = torch.stack([transform(Image.fromarray(label[ch].numpy())) for ch in range(label.shape[0])], dim=0)
    
    img = (img-img.min())/(img.max()-img.min())
    
    return img, mask, label

def test(npy_path, checkpoint_path, opt, modality='t2', K=10,masking_ratio=0.75,selected_labels=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device, opt)
    
    img, Mask, label = preprocess_npy_image(npy_path, opt.img_size, modality=modality)
    if selected_labels is None:
        Mask = Mask.to(device)
    elif len(selected_labels) > 1:
        Mask = torch.tensor(np.isin(label.cpu().numpy(), selected_labels).astype(np.uint8)).to(device)
    else:
        Mask = torch.tensor((label.cpu().numpy() == selected_labels).astype(np.uint8)).to(device)
    img = img.to(device)
    label = label.to(device)

    
    #sccsf_mask = torch.zeros_like(label,dtype=torch.int8)
    #sccsf_mask[torch.logical_or(torch.logical_or(label == 3, label == 6), label == 4)] = 1
    #sc_mask[torch.logical_or(label == 3, label == 6)] = 1
    #sccsf_mask = sccsf_mask.to(device)
    
    #sc_mask = torch.zeros_like(label,dtype=torch.int8)
    #sc_mask[torch.logical_or(label == 3, label == 6)] = 1
    #sc_mask = sc_mask.to(device)
    
    #Mask = sccsf_mask

    
    M = model.patchify(Mask) # 1,1024,64

    M_sum = torch.sum(M,dim=2) # 1,1024
    Max = M_sum.max()
    # 和mask沾边的patch也算
    patch_mask = torch.where(M_sum>0)[1]
    #patch_mask = torch.where(M_sum==Max)[1]
    patch_mask = patch_mask.cpu().numpy()
    
    if patch_mask.size <3:
        # 返回全零数组
        return np.zeros_like(img.cpu().numpy().squeeze()),np.zeros_like(img.cpu().numpy().squeeze()),np.zeros_like(img.cpu().numpy().squeeze())
        
    paste_imgs = []

    img_size = opt.img_size
    patch_size = opt.patch_size
    num_patches = (img_size // patch_size) ** 2
    
    patchs_im = np.zeros((num_patches, K, patch_size, patch_size))
    count_patch = np.zeros(num_patches)  # 计算每个 patch 被选中的次数
    count = 0
    count_threshold = 400
    
    # 要求每个mask内的patch都达到K次
    while np.any(count_patch[patch_mask] < K) and count<count_threshold:
        count = count+1

        with torch.no_grad():
            
            latent, mask, ids_restore = model.forward_encoder(img, M_sum, masking_ratio)
            x_edge, x_rec = model.forward_decoder(latent, ids_restore)
            
            _, im_masked1, im_paste = model.MAE_visualize(img, x_rec, mask)
            im_paste = im_paste.cpu().numpy()  # 转换为 numpy 数组
            
            # 获取被 mask 的 patch 序号
            masked_patch_indices = torch.nonzero(mask[0], as_tuple=False).squeeze().tolist()
            
            # 对于被选中的 patch
            for patch_id in masked_patch_indices:
                patch_id = int(patch_id)  # 确保 patch_id 是整数
                if count_patch[patch_id] < K:
                    row = patch_id // (img_size // patch_size)
                    col = patch_id % (img_size // patch_size)
                    patch = im_paste[:, row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size]
                    patchs_im[patch_id, int(count_patch[patch_id]), :, :] = patch
                    count_patch[patch_id] += 1

            paste_imgs.append(im_paste)

    print(f'Overall inferencer time is: {count}')
    
    # 计算每个 patch 的方差作为不确定性度量
    uncertainty_patches = np.var(patchs_im, axis=1)
    average_patches = np.mean(patchs_im, axis=1)
    
    # 将所有 patch 按照其位置顺序组合成一个和原图大小相同的不确定性图像
    uncertainty_img = np.zeros((img_size, img_size))
    average_img = np.zeros((img_size, img_size))

    for i in range(num_patches):
        row = i // (img_size // patch_size)
        col = i % (img_size // patch_size)
        uncertainty_img[row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size] = uncertainty_patches[i, :, :]
    
    for i in range(num_patches):
        row = i // (img_size // patch_size)
        col = i % (img_size // patch_size)
        average_img[row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size] = average_patches[i, :, :]

    original_img = img.squeeze().cpu().numpy()
    average_img[average_img==0] = original_img[average_img==0]
    
    Mask = Mask.cpu().numpy().squeeze()
    
    error_img = (original_img - average_img)
    error_img = abs(error_img)
    #error_img[error_img<0]=0
    
    uncertainty_img = uncertainty_img * Mask
    error_img = error_img*Mask
    
    return uncertainty_img,error_img,average_img


def infer_list(opt, npy_folder, checkpoint_path, task_name='spine', modality='spine', K=5, masking_ratio=0.75,selected_labels=None,output_folder=None):
    npy_list = [npy_folder + f for f in os.listdir(npy_folder) if f.endswith('.npy')]
    npy_list.sort(key=lambda x: (x.split(npy_folder)[1].split(".npy")[0]))
    
    uncertainty_output_dir = os.path.join(output_folder,'variance')
    os.makedirs(uncertainty_output_dir, exist_ok=True)
    error_output_dir = os.path.join(output_folder,'error')
    os.makedirs(error_output_dir, exist_ok=True)
    reconstruct_output_dir = os.path.join(output_folder,'reconstruct')
    os.makedirs(reconstruct_output_dir, exist_ok=True)
    
    for i, npy_path in enumerate(npy_list):
        base_name = os.path.basename(npy_path).replace('.npy', '')
        #if os.path.exists(os.path.join(output_dir, f'{base_name}.npy')):
        #    continue
        uncertainty_img, error_img, average_img = test(npy_path, checkpoint_path, opt, modality, K, masking_ratio,selected_labels)

        # 保存新的 .npy 文件
        uncertainty_npy_path = os.path.join(uncertainty_output_dir, f'{base_name}.npy')
        np.save(uncertainty_npy_path, uncertainty_img)
        
        error_npy_path = os.path.join(error_output_dir, f'{base_name}.npy')
        np.save(error_npy_path, error_img)
        
        reconstruct_npy_path = os.path.join(reconstruct_output_dir, f'{base_name}.npy')
        np.save(reconstruct_npy_path, average_img)
        
        print(f"{base_name} completed.")

if __name__ == '__main__':

    npy_folder = '/home/ubuntu/Project/MAE_dataset/spine_private/npy/SCCSF_private/less_520/train/' 
        
    opt = Pretrain_Private_Options().get_opt()
    #checkpoint_path = 'weight/SC_private/100MAE.pth' # 替换为你的模型权重路径
    task = 'SC_private'
    modality = 'spine' #'t2'
    K = 10
    masking_ratio = 0.75
    
    #对于初始化使用400个epoch来检查error和variance的变化趋势
    #先针对每个checkpoint逐一生成
    output_folder = 'test_K_10'
    
    # 获取所有的检查点文件
    checkpoints_folder = '/home/ubuntu/Project/UI-MAE_1022/finetune/weight/K_5/'  # 替换为你的模型权重文件夹路径
    checkpoint_files = [f for f in os.listdir(checkpoints_folder) if f.endswith('.pth')]
    checkpoint_files.sort()  # 如果需要，可以按字母顺序排序
    
    for checkpoint_file in checkpoint_files:
        if "200MAE" not in checkpoint_file:
            continue
        checkpoint_path = os.path.join(checkpoints_folder, checkpoint_file)
        print(f"Using checkpoint: {checkpoint_path}")
        
        output_subfolder = os.path.join(output_folder,checkpoint_file.replace('.pth',''))
        
        infer_list(opt, npy_folder, checkpoint_path, task, modality, K, masking_ratio,output_subfolder)
        
    #infer_list(opt, npy_folder, checkpoint_path, task, modality, K, masking_ratio,output_folder)
    