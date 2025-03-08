import torch
from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms
import numpy as np
import os
from utils.maeloader import *
from model.UGMAE import *
from utils.mae_visualize import *
from options import Pretrain_Private_Options
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import cv2
from scipy.stats import multivariate_normal
from scipy.ndimage import generate_binary_structure, binary_dilation, binary_erosion
from scipy.ndimage import label as Label
import torchvision.transforms.functional as F


def calculate_kl_divergence(patchs, ref_mean, ref_cov, epsilon=1e-6):
    # 将每个patch展平为一维向量
    flattened_patches = patchs.reshape(patchs.shape[0], -1)
    
    # 计算patch的均值和协方差矩阵
    patch_mean = np.mean(flattened_patches, axis=0)
    patch_cov = np.cov(flattened_patches, rowvar=False)
    
    # 添加一个小值到对角线上以确保协方差矩阵是正定的
    patch_cov += epsilon * np.eye(patch_cov.shape[0])
    
    # 创建高斯分布
    patch_dist = multivariate_normal(mean=patch_mean, cov=patch_cov)
    ref_dist = multivariate_normal(mean=ref_mean, cov=ref_cov)
    
    # 计算KL散度
    kl_div = patch_dist.entropy() - ref_dist.entropy()
    
    return kl_div



def get_percent_values(arr,per=0.3):
    # 展平数组
    flattened_arr = arr.flatten()
    
    # 计算10%的数量
    num_elements = len(flattened_arr)
    num_10_percent = int(per * num_elements)
    
    # 按升序排序
    sorted_arr = np.sort(flattened_arr)
    
    # 获取最小10%的值
    bottom_values = sorted_arr[num_10_percent]
    
    return bottom_values

def calculate_gradient(image):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = cv2.magnitude(grad_x, grad_y)
    return gradient_magnitude

def normalize_image(image):
    normalized_image = (image - image.min()) / (image.max() - image.min())
    normalized_image = (normalized_image * 255).astype(np.uint8)
    return normalized_image

def load_model(checkpoint_path, device, opt):
    model = UGMAE(img_size=opt.img_size, patch_size=opt.patch_size, embed_dim=opt.dim_encoder, depth=opt.depth, 
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

def test(npy_path, checkpoint_path, opt, modality='t2', K=10,masking_ratio=0.7):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint_path, device, opt)
    
    img, Mask, label = preprocess_npy_image(npy_path, opt.img_size, modality=modality)
    img = img.to(device)
    Mask = Mask.to(device)
    label = label.to(device)
    
    sc_mask = torch.zeros_like(label,dtype=torch.int8)
    sc_mask[torch.logical_or(label == 3, label == 6)] = 1
    sc_mask = sc_mask.to(device)
    
    # Convert to NumPy array for erosion
    sc_mask_np = sc_mask.cpu().numpy()
    #print(sc_mask_np.shape)
    # Apply erosion
    kernel = np.ones((3, 3), dtype=np.uint8)  # Define a 3x3 kernel
    if sc_mask_np.ndim == 4:
        kernel = np.expand_dims(kernel, axis=(0, 1))  # Adjust kernel for 4D input
    eroded_mask_np = binary_erosion(sc_mask_np, structure=kernel).astype(np.int8)
    
    # Convert back to tensor
    Mask = torch.tensor(eroded_mask_np, device=device, dtype=torch.int8)

    
    M = model.patchify(Mask) # 1,1024,64

    M_sum = torch.sum(M,dim=2) # 1,1024
    Max = M_sum.max()
    # 和mask沾边的patch也算
    patch_mask = torch.where(M_sum>0)[1]
    #patch_mask = torch.where(M_sum==Max)[1]
    patch_mask = patch_mask.cpu().numpy()
    
    if patch_mask.size <3:
        # 返回全零数组
        return np.zeros_like(img.cpu().numpy().squeeze()),np.zeros_like(img.cpu().numpy().squeeze()),np.zeros_like(img.cpu().numpy().squeeze()),\
            np.zeros_like(img.cpu().numpy().squeeze()),np.zeros_like(img.cpu().numpy().squeeze()),np.zeros_like(img.cpu().numpy().squeeze())
    target = model.patchify(img)
        
    paste_imgs = []
    masked_imgs = []
    masks = []
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
            
            x_rec = x_rec*M+target*(1-M)
            
            _, im_masked1, im_paste = model.MAE_visualize(img, x_rec, mask)
            im_paste = im_paste.cpu().numpy()  # 转换为 numpy 数组
            im_masked1 =im_masked1.cpu().numpy()  # 转换为 numpy 数组
            
            # 获取被 mask 的 patch 序号
            masked_patch_indices = torch.nonzero(mask[0], as_tuple=False).squeeze().tolist()
            
            # 对于被选中的 patch
            for patch_id in masked_patch_indices:
                #print(patch_id)
                #if patch_id not in patch_mask:
                #    continue
                patch_id = int(patch_id)  # 确保 patch_id 是整数
                if count_patch[patch_id] < K:
                    row = patch_id // (img_size // patch_size)
                    col = patch_id % (img_size // patch_size)
                    patch = im_paste[:, row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size]
                    patchs_im[patch_id, int(count_patch[patch_id]), :, :] = patch
                    count_patch[patch_id] += 1

            paste_imgs.append(im_paste)
            masked_imgs.append(im_masked1)
            masks.append(mask.cpu().numpy())
    print(f'Overall inferencer time is: {count}')
    
    # 计算每个 patch 的方差作为不确定性度量
    uncertainty_patches = np.var(patchs_im, axis=1)
    average_patches = np.mean(patchs_im, axis=1)
    
    # 计算每个 patch 的KL散度作为不确定性度量
    ref_mean = np.zeros(patch_size * patch_size)
    ref_cov = np.eye(patch_size * patch_size)

    kl_divergences = np.zeros(num_patches)
    for i in range(num_patches):
        kl_divergences[i] = calculate_kl_divergence(patchs_im[i], ref_mean, ref_cov)

    kl_divergences_img = np.zeros((img_size, img_size))
    for i in range(num_patches):
        row = i // (img_size // patch_size)
        col = i % (img_size // patch_size)
        kl_divergences_img[row * patch_size: (row + 1) * patch_size, col * patch_size: (col + 1) * patch_size] = kl_divergences[i]
    
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
    
    # 定义膨胀结构元素
    structure = generate_binary_structure(2, 1)
        
    # 膨胀操作
    #Mask = binary_erosion(Mask, structure=structure, iterations=2)
    error_img = error_img*Mask

    #error_img = normalize_image(error_img)
    
    return paste_imgs,masked_imgs, masks, uncertainty_img, error_img,average_img

def save_heatmap(uncertainty_img, save_path, cmap='hot'):
    plt.imshow(uncertainty_img, cmap=cmap)
    plt.colorbar()
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def apply_heatmap_to_grayscale_and_save(heatmap, image, save_path, alpha=0.4, colormap=cv2.COLORMAP_JET):
    # 确保image是float32类型
    img_gray = image.astype(np.float32)
    
    # 将灰度图像转换为三通道的彩色图像
    img_color = cv2.cvtColor(img_gray*255, cv2.COLOR_GRAY2RGB)
    
    # 确保heatmap也是float32类型
    heatmap = np.uint8(heatmap.astype(np.float32))

    heatmap = cv2.applyColorMap(heatmap, colormap)

    # 叠加热力图与原图
    superimposed_img = heatmap * alpha + img_color  # 调整因alpha叠加需要
    superimposed_img = superimposed_img / np.max(superimposed_img) * 255
    superimposed_img = np.uint8(superimposed_img)
    
    # 保存图像
    cv2.imwrite(save_path, superimposed_img)
    print(f"Image saved to {save_path}")
    
def save_combined_image(original_img, rec_img, uncertainty_img, error_img, save_path):
    # 将uncertainty_img和error_img转换为jet或hot色彩图
    uncertainty_img_color = cv2.applyColorMap(normalize_image(uncertainty_img), cv2.COLORMAP_JET)
    error_img_color = cv2.applyColorMap(normalize_image(error_img), cv2.COLORMAP_HOT)
    
    # 将灰度图original_img和rec_img扩展为三通道以便与彩色图像拼接
    original_img_3ch = cv2.cvtColor(normalize_image(original_img), cv2.COLOR_GRAY2RGB)
    rec_img_3ch = cv2.cvtColor(normalize_image(rec_img), cv2.COLOR_GRAY2RGB)
    
    # 拼接图像
    combined_img = np.concatenate((original_img_3ch, rec_img_3ch, uncertainty_img_color, error_img_color), axis=1)
    
    # 保存组合图像
    combined_img_pil = Image.fromarray(combined_img)
    combined_img_pil.save(save_path)
    print(f"Combined image saved at {save_path}")

    
def infer_case(opt, npy_path, checkpoint_path, task_name='spine', modality='spine', K=100, masking_ratio=0.7,exp_folder_path='uncertainty'):
    paste_imgs, masked_imgs, masks, uncertainty_img, error_img, average_img  \
        = test(npy_path, checkpoint_path, opt, modality, K, masking_ratio)
    
    # 创建保存图像的文件夹
    base_name = os.path.basename(npy_path).replace('.npy', '')
    output_dir = os.path.join(exp_folder_path, task_name, base_name+'_'+str(K))
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存不确定性热度图像
    uncertainty_save_path = os.path.join(output_dir, f'{base_name}_{modality}_uncertainty.png')
    save_heatmap(uncertainty_img, uncertainty_save_path, cmap='jet')

    # 保存误差热度图像
    error_save_path = os.path.join(output_dir, f'{base_name}_{modality}_error.png')
    save_heatmap(error_img, error_save_path , cmap='jet')

    # 保存不确定性覆盖的原始图像
    original_img, _, _= preprocess_npy_image(npy_path, opt.img_size, modality=modality)
    original_img =original_img.squeeze().cpu().numpy()

    uncertaintyoverlay_save_path = os.path.join(output_dir, f'{base_name}_{modality}_uncertaintyoverlay.png')
    apply_heatmap_to_grayscale_and_save(uncertainty_img, original_img, uncertaintyoverlay_save_path, alpha=0.3, colormap=cv2.COLORMAP_JET)
    
    erroroverlay_save_path = os.path.join(output_dir, f'{base_name}_{modality}_erroroverlay.png')
    apply_heatmap_to_grayscale_and_save(error_img, original_img, erroroverlay_save_path, alpha=0.3, colormap=cv2.COLORMAP_JET)

    save_ori_path = os.path.join(output_dir, f'{base_name}_{modality}_original.png')
    original_img = Image.fromarray((original_img* 255).astype(np.uint8).squeeze())
    original_img.save(save_ori_path)

    save_rec_path = os.path.join(output_dir, f'{base_name}_{modality}_reconstruct.png')
    rec_img = Image.fromarray((average_img* 255).astype(np.uint8).squeeze())
    rec_img.save(save_rec_path)

    print("Test completed. Results saved.")

def infer_list(opt, npy_folder, checkpoint_path, task_name='spine', modality='spine', K=100, masking_ratio=0.7, exp_folder_path='uncertainty'):
    npy_list = [npy_folder + f for f in os.listdir(npy_folder) if f.endswith('.npy')]
    npy_list.sort(key=lambda x: (x.split(npy_folder)[1].split(".npy")[0]))
    
    output_dir = os.path.join(exp_folder_path, task_name + f'_{modality}_{K}_{masking_ratio}')
    os.makedirs(output_dir, exist_ok=True)
    
    output_npy_dir = os.path.join(output_dir, 'npy')
    os.makedirs(output_npy_dir, exist_ok=True)
    
    error_dir = os.path.join(output_dir, 'error')
    os.makedirs(error_dir, exist_ok=True)
    
    erroroverlay_dir = os.path.join(output_dir, 'erroroverlay')
    os.makedirs(erroroverlay_dir, exist_ok=True)
    
    uncertaintyoverlay_dir = os.path.join(output_dir, 'uncertaintyoverlay')
    os.makedirs(uncertaintyoverlay_dir, exist_ok=True)
    
    uncertainty_dir = os.path.join(output_dir, 'uncertainty')
    os.makedirs(uncertainty_dir, exist_ok=True)
    
    original_dir = os.path.join(output_dir, 'original_img')
    os.makedirs(original_dir, exist_ok=True)
    
    rec_dir = os.path.join(output_dir, 'reconstruction')
    os.makedirs(rec_dir, exist_ok=True)

    # 创建文件夹保存组合图像
    combined_image_dir = os.path.join(output_dir, 'combined_images')
    os.makedirs(combined_image_dir, exist_ok=True)
    
    for i, npy_path in enumerate(npy_list):
        paste_imgs, masked_imgs, masks, uncertainty_img, error_img, average_img = test(npy_path, checkpoint_path, opt, modality, K, masking_ratio)
        base_name = os.path.basename(npy_path).replace('.npy', '')
        
        # 保存不确定性图像
        uncertainty_save_path = os.path.join(uncertainty_dir, f'{base_name}_uncertainty.png')
        save_heatmap(uncertainty_img, uncertainty_save_path, cmap='jet')
        
        # 保存误差图像
        error_save_path = os.path.join(error_dir, f'{base_name}_error.png')
        save_heatmap(error_img, error_save_path, cmap='jet')
        
        # 加载并处理原始图像
        original_img, _, _ = preprocess_npy_image(npy_path, opt.img_size, modality=modality)
        original_img = original_img.squeeze().cpu().numpy()

        # 保存叠加的错误和不确定性图像
        erroroverlay_save_path = os.path.join(erroroverlay_dir, f'{base_name}_erroroverlay.png')
        apply_heatmap_to_grayscale_and_save(error_img, original_img, erroroverlay_save_path, alpha=0.3, colormap=cv2.COLORMAP_JET)

        uncertaintyoverlay_save_path = os.path.join(uncertaintyoverlay_dir, f'{base_name}_uncertaintyoverlay.png')
        apply_heatmap_to_grayscale_and_save(uncertainty_img, original_img, uncertaintyoverlay_save_path, alpha=0.3, colormap=cv2.COLORMAP_JET)
        
        # 保存原始和重建图像
        save_ori_path = os.path.join(original_dir, f'{base_name}_original.png')
        original_img_pil = Image.fromarray((original_img * 255).astype(np.uint8).squeeze())
        original_img_pil.save(save_ori_path)
        
        save_rec_path = os.path.join(rec_dir, f'{base_name}_reconstruct.png')
        rec_img = Image.fromarray((average_img * 255).astype(np.uint8).squeeze())
        rec_img.save(save_rec_path)
        
        # 保存组合图像
        combined_image_path = os.path.join(combined_image_dir, f'{base_name}_combined.png')
        save_combined_image(original_img, average_img, uncertainty_img, error_img, combined_image_path)
        
        # 处理并保存新的 .npy 文件
        original_npy = np.load(npy_path)
        uncertainty_img = uncertainty_img[np.newaxis, :, :]
        error_img = error_img[np.newaxis, :, :]
        average_img = average_img[np.newaxis, :, :]
        
        new_npy = np.concatenate((original_npy, uncertainty_img, error_img, average_img), axis=0)
        new_npy_path = os.path.join(output_npy_dir, f'{base_name}.npy')
        np.save(new_npy_path, new_npy)
        
        print(f"{base_name} completed.")
        

if __name__ == '__main__':
    
    #是否对单个case进行推断
    case=False
    
    if case:
        npy_path = './data/private/train/13.npy'
        
        opt = Pretrain_Private_Options().get_opt()
        checkpoint_path = 'weight/SC_private/100MAE.pth' # 替换为你的模型权重路径
        task = 'SC_1'
        modality = 'spine' #'t2'
        K = 100
        masking_ratio = 0.7
        exp_name = 'uncertainty'
        
        infer_case(opt, npy_path, checkpoint_path, task,modality, K, masking_ratio,exp_name)
        
    else:
        npy_folder = './data/private/train/' 
        
        opt = Pretrain_Private_Options().get_opt()
        checkpoint_path = './weight/adaptation_private/test/MAE.pth' # 替换为你的模型权重路径
        task = 'SC_private'
        modality = 'spine' #'t2'
        K = 5
        masking_ratio = 0.75
        exp_name = './uncertainty'
        
        infer_list(opt, npy_folder, checkpoint_path, task, modality, K, masking_ratio,exp_name)
    