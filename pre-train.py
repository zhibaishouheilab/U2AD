import torch
import logging
from utils.maeloader import *
from model.UGMAE import *
from utils.mae_visualize import *
from options import Pretrain_Private_Options
from utils.variance_generate import infer_list
from utils.variance_ratio import calculate_average_summation
import os
from tensorboardX import SummaryWriter

def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = init_lr * decay
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
opt = Pretrain_Private_Options().get_opt()

mae = UGMAE(img_size=opt.img_size,patch_size=opt.patch_size, embed_dim=opt.dim_encoder, depth=opt.depth, num_heads=opt.num_heads, in_chans=1,
        decoder_embed_dim=opt.dim_decoder, decoder_depth=opt.decoder_depth, decoder_num_heads=opt.decoder_num_heads,
        mlp_ratio=opt.mlp_ratio,norm_pix_loss=False,patchwise_loss=opt.use_patchwise_loss)

os.makedirs(opt.img_save_path,exist_ok=True)
os.makedirs(opt.weight_save_path,exist_ok=True)

total_epoch = 0
# 每次迭代的训练轮数是10 epoch
iterative_epoch = opt.iterative_epoch 

variance_folder = './variance/pretrain/baseline'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_loader = get_maeloader(batchsize=opt.batch_size, shuffle=True,pin_memory=True,img_size=opt.img_size,
            img_root=opt.data_root,num_workers=opt.num_workers,augment=opt.augment,modality=opt.modality)

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

writer = SummaryWriter(log_dir='./logs/pretrain/baseline')

#0epoch记录variance和error
save_folder_variance = os.path.join(variance_folder,f'epoch_{str(total_epoch)}','variance')
variance_average = calculate_average_summation(save_folder_variance)
save_folder_error = os.path.join(variance_folder,f'epoch_{str(total_epoch)}','error')
error_average = calculate_average_summation(save_folder_error)
writer.add_scalar('Variance', variance_average, total_epoch)
writer.add_scalar('Error', error_average, total_epoch)

while (total_epoch<opt.epoch):
    for epoch in range(total_epoch,total_epoch+iterative_epoch):
        for i,(img,label) in enumerate(train_loader):

            adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        
            optimizer.zero_grad()

            img = img.to(device,dtype=torch.float)
            label = label.to(device)
            #print("now image shape is:", img.shape)

            rec_loss, edge_loss,edge_gt,x_edge,x_rec,mask = mae(img,label,opt.masking_ratio,epoch)
            loss = rec_loss * opt.l1_loss + edge_loss 
        
            loss.backward()
            optimizer.step()

            print(
                    "[Epoch %d/%d] [Batch %d/%d] [rec_loss: %f] [edge_loss: %f] [lr: %f]"
                    % (epoch, opt.epoch, i, len(train_loader), rec_loss.item(),edge_loss.item(),get_lr(optimizer))
                )

            if i % opt.save_output == 0:
                logging.info("[Epoch %d/%d] [Batch %d/%d] [rec_loss: %f] [edge_loss: %f] [lr: %f]"
                    % (epoch, opt.epoch, i, len(train_loader), rec_loss.item(),edge_loss.item(),get_lr(optimizer)))

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
