# add image at successive layer

import os
from config import Config 
opt = Config('training.yml')

gpus = ','.join([str(i) for i in opt.GPU])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus

import torch
torch.backends.cudnn.benchmark = True

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import random
import time
import sys
import numpy as np
from skimage import img_as_ubyte
import gc

import utils
from data_RGB import get_training_data, get_validation_data
from RMOBF import RMOBF
import losses
from tqdm import tqdm
from pdb import set_trace as stx
from multiprocessing import freeze_support
# from torchmetrics import StructuralSimilarityIndexMeasure
import cv2


random_seed = 1234
torch.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU


start_epoch = 1
mode = opt.MODEL.MODE

result_dir = os.path.join(opt.TRAINING.SAVE_DIR,  'results')
model_dir  = os.path.join(opt.TRAINING.SAVE_DIR,  'models')

utils.mkdir(result_dir)
utils.mkdir(model_dir)

train_dir = opt.TRAINING.TRAIN_DIR
valid_dir   = opt.TRAINING.VAL_DIR
test_dir   = opt.TESTING.TEST_DIR

######### Model ###########
model_restoration = RMOBF()


device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
  print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

print(opt)
new_lr = opt.OPTIM.LR_INITIAL

optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999),eps=1e-8)



# print(model_restoration)
model_restoration.cuda()

######### Resume ###########
if opt.TRAINING.RESUME:
    path_chk_rest    = utils.get_last_path(model_dir, '_latest.pth')
    utils.load_checkpoint(model_restoration,path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------------------')


######### Loss ###########
criterion_char = losses.CharbonnierLoss()
criterion_perp = losses.perceptual_loss()


# ssim = StructuralSimilarityIndexMeasure()

######### DataLoaders ###########
train_dataset = get_training_data(train_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True, drop_last=False, num_workers=8, pin_memory=True)

valid_dataset = get_validation_data(valid_dir, {'patch_size':opt.TRAINING.TRAIN_PS})
valid_loader = DataLoader(dataset=valid_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=8, pin_memory=True)


print('===> Start Epoch {} End Epoch {}'.format(start_epoch,opt.OPTIM.NUM_EPOCHS + 1))
print('===> Loading datasets')

best_psnr = 0
best_epoch = 0

min_valid_loss = 1000


if __name__ == "__main__":
    freeze_support()
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        train_id = 1
        loss = 0
        loss_char = 0
        loss_perceptual = 0
        model_restoration.train()
        for i, data in enumerate(train_loader, 0):

            # zero_grad
            for param in model_restoration.parameters():
                param.grad = None

            
            input_ = data[0].cuda()
            target = data[1].cuda()
            restored = model_restoration(input_)

            # Compute loss at each stage

            loss_char = criterion_char(restored, target)
            loss_perceptual = criterion_perp(restored, target)

            loss = loss_char + loss_perceptual

            loss.backward()
            optimizer.step()
            epoch_loss +=loss.detach().item()

            if i % 10 == 0:
                print(f'epoch : {epoch}, iter : {i}, loss_char : {loss_char:.5f}, loss_percept : {loss_perceptual:.5f}, loss : {loss.item():.5f}')

        print("------------------------------------------------------------------")
        print("TRANING Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}".format(epoch, time.time()-epoch_start_time, epoch_loss / len(train_loader)))
        print("------------------------------------------------------------------")

        restored_img = torch.clamp(restored, 0, 1)
        restored_img = restored_img.permute(0, 2, 3, 1).cpu().detach().numpy()
        print(restored_img[0].shape)
        restored_img = img_as_ubyte(restored_img[0])
        cv2.imwrite(os.path.join(result_dir, "epoch"+ str(epoch) + "_" + data[2][0] + '.png'),cv2.cvtColor(restored_img, cv2.COLOR_RGB2BGR))

        if epoch > 200:

            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer' : optimizer.state_dict()
                        }, os.path.join(model_dir,f"model_epoch_{epoch}.pth"))

            # del val_dataset, val_loader, models, psnr_val_rgb, ssim_val_rgb, data_val, target, input_, filenames, restored, res, tar, restored_img
        torch.cuda.empty_cache()

                #### Evaluation ####
        if epoch%opt.TRAINING.VAL_AFTER_EVERY == 0:
            valid_loss = 0
            model_restoration.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate(tqdm(valid_loader), 0):
                val_input_ = data_val[0].cuda()
                val_target = data_val[1].cuda()
            
                with torch.no_grad():
                    val_restored = model_restoration(val_input_)

                    val_loss_char = criterion_char(val_restored, val_target)
                    val_loss_perceptual = criterion_perp(val_restored, val_target)

                    val_loss = val_loss_char + val_loss_perceptual
                    valid_loss += val_loss
                    
                val_restored = val_restored[0]

                for res,tar in zip(val_restored,val_target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))

            psnr_val_rgb  = torch.stack(psnr_val_rgb).mean().item()

            if valid_loss < min_valid_loss:
                torch.save({'epoch': epoch, 
                'state_dict': model_restoration.state_dict(),
                'optimizer' : optimizer.state_dict()
                }, os.path.join(model_dir,"model_best.pth"))

            # if psnr_val_rgb > best_psnr:
            #     best_psnr = psnr_val_rgb
            #     best_epoch = epoch
            #     torch.save({'epoch': epoch, 
            #                 'state_dict': model_restoration.state_dict(),
            #                 'optimizer' : optimizer.state_dict()
            #                 }, os.path.join(model_dir,"model_best.pth"))


            print()
            print("---------------------------VALIDATION-----------------------------")
            print("VALIDATION Epoch: {}\tLoss: {:.4f}".format(epoch, valid_loss / len(valid_loader)))
            print("------------------------------------------------------------------")

            del valid_loss, val_input_, val_target, val_restored, val_loss, val_loss_char, val_loss_perceptual, res, tar
            torch.cuda.empty_cache()
            print()
            





