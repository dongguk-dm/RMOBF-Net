
from cv2 import CAP_PVAPI_PIXELFORMAT_BGR24
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx


class Resblock(nn.Module):
    def __init__(self, channels, ksize):
        super(Resblock, self).__init__()
        self.res_conv1 = nn.Conv2d(channels, channels, kernel_size=ksize,  stride = 1, padding = ksize // 2, bias = False)
        self.PReLU = nn.PReLU()
        self.res_conv2 = nn.Conv2d(channels, channels, kernel_size=ksize,  stride = 1, padding = ksize // 2, bias = False)

    def forward(self, x):
        out = self.res_conv1(x)
        out = self.PReLU(x)
        out = self.res_conv2(x)
        return x + out

##########################################################################
class RMOBF(nn.Module):
    def __init__(self, in_c=3, out_feat=48, kernel_size=3, stride = 1, bias=False):
        super(RMOBF, self).__init__()
        
        self.low_feat1 = nn.Sequential(nn.Conv2d(in_c, out_feat, kernel_size, stride, padding = (kernel_size//2), bias = bias), nn.PReLU())
        self.res_block256_1 = nn.Sequential(*[Resblock(out_feat, kernel_size) for _ in range(3)])

        # ENCODER PART -> apply bellow method 2 times (ori res * 1/4)
        self.downsample128 = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False), # down
                                        nn.Conv2d(out_feat, out_feat * 2, kernel_size, stride=1, padding=(kernel_size//2), bias=bias), # increase C
                                        nn.PReLU()) 

        self.down_block128 = nn.Sequential(*[Resblock(out_feat * 2, kernel_size) for _ in range(3)])

        self.downsample64 = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False), # down
                                        nn.Conv2d(out_feat * 2, out_feat * 4, kernel_size, stride=1, padding=(kernel_size//2), bias=bias), # increase C
                                        nn.PReLU()) 
        
        self.down_block64 = nn.Sequential(*[Resblock(out_feat * 4, kernel_size) for _ in range(3)])

        self.downsample32 = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False), # down
                                        nn.Conv2d(out_feat * 4, out_feat * 6, kernel_size, stride=1, padding=(kernel_size//2), bias=bias), # increase C
                                        nn.PReLU()) 
        
        self.down_block32 = nn.Sequential(*[Resblock(out_feat * 6, kernel_size) for _ in range(3)])

        self.downsample16 = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False), # down
                                        nn.Conv2d(out_feat * 6, out_feat * 8, kernel_size, stride=1, padding=(kernel_size//2), bias=bias), # increase C
                                        nn.PReLU()) 
        
        self.down_block16 = nn.Sequential(*[Resblock(out_feat * 8, kernel_size) for _ in range(3)])

        self.bottom_down8 = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False), # down
                                        nn.Conv2d(out_feat * 8, out_feat * 10, kernel_size, stride=1, padding=(kernel_size//2), bias=bias), # increase C
                                        nn.PReLU()) 
        
        self.bottom_block8 = nn.Sequential(*[Resblock(out_feat * 10, kernel_size) for _ in range(3)])


        # DeCODER PART -> apply bellow method 2 times (ori res)

        self.bottom_up16 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # down
                                      nn.Conv2d(out_feat * 10, out_feat * 8, kernel_size, stride=1, padding=(kernel_size//2), bias=bias), # increase C
                                      nn.PReLU()) 

        self.conv16 = nn.Conv2d(out_feat * 16, out_feat * 8, kernel_size=1, stride = 1, padding = 0, bias=bias)        

        self.up_block16 = nn.Sequential(*[Resblock(out_feat * 8, kernel_size) for _ in range(3)])

        self.upsample32 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # down
                                      nn.Conv2d(out_feat * 8, out_feat * 6, kernel_size, stride=1, padding=(kernel_size//2), bias=bias), # increase C
                                      nn.PReLU()) 


        self.conv32 = nn.Conv2d(out_feat * 12, out_feat * 6, kernel_size=1, stride = 1, padding = 0, bias=bias)
        self.up_block32 = nn.Sequential(*[Resblock(out_feat * 6, kernel_size) for _ in range(3)])



        self.upsample64 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # down
                                      nn.Conv2d(out_feat * 6, out_feat * 4, kernel_size, stride=1, padding=(kernel_size//2), bias=bias), # increase C
                                      nn.PReLU()) 
        self.conv64 = nn.Conv2d(out_feat * 8, out_feat * 4, kernel_size=1, stride = 1, padding = 0, bias=bias)
        self.up_block64 = nn.Sequential(*[Resblock(out_feat * 4, kernel_size) for _ in range(3)])

        self.upsample128 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # down
                                      nn.Conv2d(out_feat * 4, out_feat * 2, kernel_size, stride=1, padding=(kernel_size//2), bias=bias), # increase C
                                      nn.PReLU()) 
        self.conv128 = nn.Conv2d(out_feat * 4, out_feat * 2, kernel_size=1, stride = 1, padding = 0, bias=bias)
        self.up_block128 = nn.Sequential(*[Resblock(out_feat * 2, kernel_size) for _ in range(3)])

        

        self.upsample256 = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), # down
                                      nn.Conv2d(out_feat * 2, out_feat, kernel_size, stride=1, padding=(kernel_size//2), bias=bias), # increase C
                                      nn.PReLU()) 

        self.conv256 = nn.Conv2d(out_feat * 2, out_feat, kernel_size=1, stride = 1, padding = 0, bias=bias)
        self.res_block256_2 = nn.Sequential(*[Resblock(out_feat, kernel_size) for _ in range(3)])
        
        # self.res_block
        

        self.last_feat = nn.Conv2d(out_feat, in_c, kernel_size, stride, padding = (kernel_size//2), bias = bias)



    def forward(self, blur_img):

        
        # stage 1
        low_feat = self.low_feat1(blur_img)
        enc_256 = self.res_block256_1(low_feat)

        down_128 = self.downsample128(enc_256)
        enc_128 = self.down_block128(down_128)

        down_64 = self.downsample64(enc_128)
        enc_64 = self.down_block64(down_64)
        
        down_32 = self.downsample32(enc_64)
        enc_32 = self.down_block32(down_32)

        down_16 = self.downsample16(enc_32)
        enc_16 = self.down_block16(down_16)
        
        
        down_8 = self.bottom_down8(enc_16)
        
        bottom = self.bottom_block8(down_8)
        
        up_16 = self.bottom_up16(bottom)

        
        cat_16 = torch.cat((up_16, enc_16), dim=1)

        cat_16 = self.conv16(cat_16)
        dec_16 = self.up_block16(cat_16)
        up_32 = self.upsample32(dec_16)

        cat_32 = torch.cat((up_32, enc_32), dim=1)
        cat_32 = self.conv32(cat_32)
        dec_32 = self.up_block32(cat_32)
        up_64 = self.upsample64(dec_32)

        cat_64 = torch.cat((up_64, enc_64), dim=1)

        cat_64 = self.conv64(cat_64)
        dec_64 = self.up_block64(cat_64)
        up_128 = self.upsample128(dec_64)

        cat_128 = torch.cat((up_128, enc_128), dim=1)
        cat_128 = self.conv128(cat_128)
        dec_128 = self.up_block128(cat_128)
        up_256 = self.upsample256(dec_128)

        
        cat_256 = torch.cat((up_256, enc_256), dim=1)
        cat_256 = self.conv256(cat_256)
        dec_256 = self.res_block256_2(cat_256)

        out = self.last_feat(dec_256)

        return out



        

       