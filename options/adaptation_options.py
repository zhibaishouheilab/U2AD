import argparse
import os

class Adaptation_Private_Options():
    def __init__(self):

        self.parser = argparse.ArgumentParser()
        self.parser.add_argument('--lr', type=float, default=3e-3, help='learning rate')
        self.parser.add_argument("--batch_size", default=10, type=int)
        self.parser.add_argument("--epoch", default=200, type=int) #原来是100
        self.parser.add_argument("--patch_size", default=8, type=int)
        self.parser.add_argument("--img_size", default=256, type=int)
        self.parser.add_argument("--decay_epoch", default=50, type=float)
        self.parser.add_argument("--decay_rate", default=0.1, type=float)
        self.parser.add_argument("--l1_loss", default=10, type=int) # 原来是10
        self.parser.add_argument("--augment", default=True) #preform data augmentation
        self.parser.add_argument("--modality", default='spine') #using all modalities for pre-training (t1, t2, t1c, flair)
        self.parser.add_argument("--masking_ratio", default=0.75,type=float) 
        self.parser.add_argument("--num_workers", default=36, type=int) #36
        
        self.parser.add_argument('--use_checkpoints', default=True)
        self.parser.add_argument('--img_save_path', type=str,default='./snapshot/adaptation_private/test/')
        self.parser.add_argument('--weight_save_path', type=str,default='./weight/adaptation_private/test/')
        self.parser.add_argument('--checkpoint_path', type=str,default='./weight/pretrain/baseline/MAE.pth')
        self.parser.add_argument("--data_root", default='../data/private/train/')

        self.parser.add_argument("--depth", default=12, type=int)
        self.parser.add_argument("--use_patchwise_loss", default=True)
        self.parser.add_argument("--decoder_depth", default=8, type=int)
        self.parser.add_argument("--save_output", default=200, type=int)
        self.parser.add_argument("--iterative_epoch", default=10, type=int)
        self.parser.add_argument("--num_heads", default=16, type=int)
        self.parser.add_argument("--decoder_num_heads", default=8, type=int)
        self.parser.add_argument("--mlp_ratio", default=4, type=int)
        self.parser.add_argument("--dim_encoder", default=128, type=int)
        self.parser.add_argument("--dim_decoder", default=64, type=int)
        self.parser.add_argument("--log_path", default='./log/adaptation_private/test.log')

    def get_opt(self):
        self.opt = self.parser.parse_args()
        return self.opt

        
    def get_opt(self):
        self.opt = self.parser.parse_args()
        return self.opt

        
