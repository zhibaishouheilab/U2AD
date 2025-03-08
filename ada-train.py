import torch
import logging
from utils.maeloader import *
from U2AD.model.UGMAE import *
from utils.mae_visualize import *
from utils.variance_generate import infer_list
from utils.variance_ratio import calculate_average_summation
from options import Adaptation_Private_Options
import os
from tensorboardX import SummaryWriter



import cv2


def process_error(error_map, threshold_sig=90, threshold_size=2):
    # 过滤掉大于90%阈值的误差区域
    non_zero_values = error_map[error_map != 0]
    
    if len(non_zero_values) == 0:
        print("No non-zero values in the error map")
        # If there are no non-zero values, return an empty filtered error map
        return np.zeros_like(error_map)
    
    threshold_error = np.percentile(non_zero_values, threshold_sig)
    error_map_filtered = np.where(error_map > threshold_error, error_map, 0)

    # 连通域分析并去除小于阈值的连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((error_map_filtered > 0).astype(np.uint8))
    
    # 存储每个连通域的平均误差值、总误差值、以及乘积
    components_info = []
    for i in range(1, num_labels):
        component_mask = (labels == i)
        component_values = error_map_filtered[component_mask]
        if stats[i, cv2.CC_STAT_AREA] >= threshold_size:
            avg_error = np.mean(component_values)
            total_error = np.sum(component_values)
            score = avg_error * total_error  # 计算乘积
            
            components_info.append((i, score, avg_error, total_error))
    
    # 根据得分进行排序，保留前3个连通域
    components_info.sort(key=lambda x: x[1], reverse=True)  # 根据乘积排序
    selected_components = components_info[:3] if len(components_info) > 3 else components_info
    
    # 创建新的过滤后的错误图，保留前三个连通域
    filtered_error_map = np.zeros_like(error_map_filtered)
    for comp_info in selected_components:
        comp_idx = comp_info[0]
        filtered_error_map[labels == comp_idx] = error_map_filtered[labels == comp_idx]
            
    return filtered_error_map


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
opt = Adaptation_Private_Options().get_opt()

mae = UGMAE(img_size=opt.img_size,patch_size=opt.patch_size, embed_dim=opt.dim_encoder, depth=opt.depth, num_heads=opt.num_heads, in_chans=1,
        decoder_embed_dim=opt.dim_decoder, decoder_depth=opt.decoder_depth, decoder_num_heads=opt.decoder_num_heads,
        mlp_ratio=opt.mlp_ratio,norm_pix_loss=False,patchwise_loss=opt.use_patchwise_loss)

os.makedirs(opt.img_save_path,exist_ok=True)
os.makedirs(opt.weight_save_path,exist_ok=True)

total_epoch = 0
# 每次迭代的训练轮数是10 epoch
iterative_epoch = opt.iterative_epoch

variance_folder = './variance/adaptation_private/test'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

optimizer = torch.optim.Adam(mae.parameters(), lr=opt.lr,betas=(0.9, 0.95))
mae = mae.to(device)

if opt.use_checkpoints == True:
    print('load checkpoint......',opt.checkpoint_path)
    mae.load_state_dict(torch.load(opt.checkpoint_path, map_location=torch.device(device)),strict=False)
    
logging.basicConfig(filename=opt.log_path,
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

output_folder = os.path.join(variance_folder,f'epoch_{str(total_epoch)}')
checkpoint_path = opt.checkpoint_path
#先记录第0epoch
torch.save(mae.state_dict(), opt.weight_save_path + str(total_epoch) + 'MAE.pth')
# 0epoch生成variance和error
infer_list(opt,opt.data_root,checkpoint_path,K=10,masking_ratio=0.75,selected_labels=[3,4,6],output_folder=output_folder)


# tau是温度系数，用于控制方差对比情况
tau=1

writer = SummaryWriter(log_dir='./logs/adaptation_private/test')
#0epoch记录variance和error
save_folder_variance = os.path.join(variance_folder,f'epoch_{str(total_epoch)}','variance')
variance_average = calculate_average_summation(save_folder_variance)
save_folder_error = os.path.join(variance_folder,f'epoch_{str(total_epoch)}','error')
error_average = calculate_average_summation(save_folder_error)

writer.add_scalar('Variance', variance_average, total_epoch)
writer.add_scalar('Error', error_average, total_epoch)



while (total_epoch<150):
    variance_path = os.path.join(variance_folder,f'epoch_{str(total_epoch)}','variance')
    error_path = os.path.join(variance_folder,f'epoch_{str(total_epoch)}','error')
    train_loader = get_maeloader(batchsize=opt.batch_size, shuffle=True,pin_memory=True,img_size=opt.img_size,
        img_root=opt.data_root,num_workers=opt.num_workers,augment=opt.augment,modality=opt.modality,variance_path=variance_path)
    for epoch in range(total_epoch,total_epoch+iterative_epoch):
        for i,(img,label,variance) in enumerate(train_loader):

            adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        
            optimizer.zero_grad()

            img = img.to(device,dtype=torch.float)
            variance = variance.to(device,dtype=torch.float)
            label = label.to(device)
            rec_loss, edge_loss,edge_gt,x_edge,x_rec,mask = mae(img,label,opt.masking_ratio,epoch,variance,tau)
            
            # Calculate the total loss
            loss = rec_loss * opt.l1_loss + edge_loss
        
            loss.backward()
            optimizer.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [rec_loss: %f] [edge_loss: %f] [lr: %f]"
                % (epoch, opt.epoch, i, len(train_loader), rec_loss.item(),edge_loss.item(),get_lr(optimizer))
                )

            if i % opt.save_output == 0:
                y1, im_masked1, im_paste1 = mae.MAE_visualize(img, x_rec, mask)
                y2, im_masked2, im_paste2 = mae.MAE_visualize(edge_gt, x_edge, mask)
                edge_gt,img = edge_gt.cpu(),img.cpu()
                #print(img[0].max())
                save_image([img[0],im_masked1,im_paste1,edge_gt[0],im_masked2,im_paste2],
                    opt.img_save_path + str(epoch) + ' ' + str(i)+'.png', nrow=3,normalize=False)
                logging.info("[Epoch %d/%d] [Batch %d/%d] [rec_loss: %f] [edge_loss: %f] [lr: %f]"
                    % (epoch, opt.epoch, i, len(train_loader), rec_loss.item(),edge_loss.item(),get_lr(optimizer)))

        print(epoch+1)
        if (epoch+1) % opt.iterative_epoch == 0:
            torch.save(mae.state_dict(), opt.weight_save_path + str(epoch+1) + 'MAE.pth')
    
    total_epoch+=iterative_epoch
    output_folder = os.path.join(variance_folder,f'epoch_{str(total_epoch)}')
    checkpoint_path = opt.weight_save_path + str(total_epoch) + 'MAE.pth'
    infer_list(opt,opt.data_root,checkpoint_path,K=10,masking_ratio=0.75,selected_labels=[3,4,6],output_folder=output_folder)

    save_folder_variance = os.path.join(variance_folder,f'epoch_{str(total_epoch)}','variance')
    variance_average = calculate_average_summation(save_folder_variance)
    
    save_folder_error = os.path.join(variance_folder,f'epoch_{str(total_epoch)}','error')
    error_average = calculate_average_summation(save_folder_error)


    writer.add_scalar('Variance', variance_average, total_epoch)
    writer.add_scalar('Error', error_average, total_epoch)

    print("[Variance: %f] [Error: %f]" % (variance_average, error_average))
    logging.info("[Variance: %f] [Error: %f]" % (variance_average, error_average))


while (total_epoch<opt.epoch):
    variance_path = os.path.join(variance_folder,f'epoch_{str(total_epoch)}','variance')
    error_path = os.path.join(variance_folder,f'epoch_{str(total_epoch)}','error')
    train_loader = get_maeloader(batchsize=opt.batch_size, shuffle=True,pin_memory=True,img_size=opt.img_size,
        img_root=opt.data_root,num_workers=opt.num_workers,augment=opt.augment,modality=opt.modality,variance_path=variance_path,error_path=error_path)
    for epoch in range(total_epoch,total_epoch+iterative_epoch):
        for i,(img,label,variance,error) in enumerate(train_loader):

            adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        
            optimizer.zero_grad()

            img = img.to(device,dtype=torch.float)
            variance = variance.to(device,dtype=torch.float)
            
            for i, error_map in enumerate(error):
                error_filtered = process_error(error_map.squeeze(),threshold_sig=90,threshold_size=2)
                error[i,0,:,:] = torch.from_numpy(error_filtered)
            error = error.to(device,dtype=torch.float)

            label = label.to(device)
            rec_loss, edge_loss,edge_gt,x_edge,x_rec,mask = mae(img,label,opt.masking_ratio,epoch,variance,tau,error)
            
            # Calculate the total loss
            loss = rec_loss * opt.l1_loss + edge_loss
        
            loss.backward()
            optimizer.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [rec_loss: %f] [edge_loss: %f] [lr: %f]"
                % (epoch, opt.epoch, i, len(train_loader), rec_loss.item(),edge_loss.item(),get_lr(optimizer))
                )

            if i % opt.save_output == 0:
                y1, im_masked1, im_paste1 = mae.MAE_visualize(img, x_rec, mask)
                y2, im_masked2, im_paste2 = mae.MAE_visualize(edge_gt, x_edge, mask)
                edge_gt,img = edge_gt.cpu(),img.cpu()
                #print(img[0].max())
                save_image([img[0],im_masked1,im_paste1,edge_gt[0],im_masked2,im_paste2],
                    opt.img_save_path + str(epoch) + ' ' + str(i)+'.png', nrow=3,normalize=False)
                logging.info("[Epoch %d/%d] [Batch %d/%d] [rec_loss: %f] [edge_loss: %f] [lr: %f]"
                    % (epoch, opt.epoch, i, len(train_loader), rec_loss.item(),edge_loss.item(),get_lr(optimizer)))

        print(epoch+1)
        if (epoch+1) % opt.iterative_epoch == 0:
            torch.save(mae.state_dict(), opt.weight_save_path + str(epoch+1) + 'MAE.pth')
    
    total_epoch+=iterative_epoch
    output_folder = os.path.join(variance_folder,f'epoch_{str(total_epoch)}')
    checkpoint_path = opt.weight_save_path + str(total_epoch) + 'MAE.pth'
    infer_list(opt,opt.data_root,checkpoint_path,K=10,masking_ratio=0.75,selected_labels=[3,4,6],output_folder=output_folder)

    save_folder_variance = os.path.join(variance_folder,f'epoch_{str(total_epoch)}','variance')
    variance_average = calculate_average_summation(save_folder_variance)
    
    save_folder_error = os.path.join(variance_folder,f'epoch_{str(total_epoch)}','error')
    error_average = calculate_average_summation(save_folder_error)

    writer.add_scalar('Variance', variance_average, total_epoch)
    writer.add_scalar('Error', error_average, total_epoch)

    print("[Variance: %f] [Error: %f]" % (variance_average, error_average))
    logging.info("[Variance: %f] [Error: %f]" % (variance_average, error_average))
    
writer.close()

torch.save(mae.state_dict(), opt.weight_save_path + './MAE.pth')
