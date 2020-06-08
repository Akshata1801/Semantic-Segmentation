# -*- coding: utf-8 -*-
"""
Created on Mon May 11 11:49:31 2020

@author: akpo2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class segnet(nn.Module):
    def __init__(self):
        super(segnet,self).__init__()
        
        batchNorm_momentum = 0.1
        n_classes = 34
        
        self.conv10 = nn.Conv2d(3,64,kernel_size=3,stride=1 ,padding=(1,1))
        self.batch_norm10 = nn.BatchNorm2d(64,affine=False)
        self.conv11 = nn.Conv2d(64,64, kernel_size=3, padding=1)
        self.batch_norm11 = nn.BatchNorm2d(64, momentum=batchNorm_momentum, affine=False)
        
        self.conv20 = nn.Conv2d(64,128, kernel_size=3,stride=1, padding=(1,1))
        self.batch_norm20 = nn.BatchNorm2d(128,momentum=batchNorm_momentum,affine=False)
        self.conv21 = nn.Conv2d(128,128,kernel_size=3,stride=1, padding=(1,1))
        self.batch_norm21 = nn.BatchNorm2d(128, momentum=batchNorm_momentum,affine=False)
        
        self.conv30 = nn.Conv2d(128,256, kernel_size=3,stride=1, padding=1)
        self.batch_norm30 = nn.BatchNorm2d(256,momentum=batchNorm_momentum, affine=False)
        self.conv31 = nn.Conv2d(256,256, kernel_size=3,stride=1, padding=1)
        self.batch_norm31 = nn.BatchNorm2d(256,momentum=batchNorm_momentum,affine=False)
        self.conv32 = nn.Conv2d(256,256, kernel_size=3,stride=1,padding=1)
        self.batch_norm32 = nn.BatchNorm2d(256, momentum=batchNorm_momentum, affine=False)
        
        self.conv40 = nn.Conv2d(256,512, kernel_size=3, padding=1)
        self.batch_norm40 = nn.BatchNorm2d(512, momentum=batchNorm_momentum,affine=False)
        self.conv41 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch_norm41 = nn.BatchNorm2d(512, momentum=batchNorm_momentum, affine=False)
        self.conv42 = nn.Conv2d(512,512, kernel_size=3,padding=1)
        self.batch_norm42 = nn.BatchNorm2d(512, momentum=batchNorm_momentum,affine=False)
        
        self.conv50 = nn.Conv2d(512, 512, kernel_size=3,padding=1)
        self.batch_norm50 = nn.BatchNorm2d(512,512,momentum=batchNorm_momentum,affine=False)
        self.conv51 = nn.Conv2d(512,512,kernel_size=3 , padding=1)
        self.batch_norm51 = nn.BatchNorm2d(512, momentum=batchNorm_momentum,affine=False)
        self.conv52 = nn.Conv2d(512,512, kernel_size=3, padding=1)
        self.batch_norm52 = nn.BatchNorm2d(512,momentum=batchNorm_momentum,affine=False)
        
        ###### DEcoder
        
        self.conv62 = nn.Conv2d(512,512, kernel_size=3, padding=1)
        self.batch_norm62 = nn.BatchNorm2d(512, momentum=batchNorm_momentum, affine=False)
        self.conv61 = nn.Conv2d(512,512, kernel_size=3, padding=1)
        self.batch_norm61 = nn.BatchNorm2d(512, momentum=batchNorm_momentum, affine=False)
        self.conv60 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch_norm60 = nn.BatchNorm2d(512, momentum=batchNorm_momentum,affine=False)
        
        self.conv72 = nn.Conv2d(512,512, kernel_size=3,stride=1,padding=1)
        self.batch_norm72 = nn.BatchNorm2d(512, momentum=batchNorm_momentum, affine=False)
        self.conv71 = nn.Conv2d(512, 512, kernel_size=3,stride=1, padding=1)
        self.batch_norm71 = nn.BatchNorm2d(512, momentum=batchNorm_momentum, affine=False)
        self.conv70 = nn.Conv2d(512,256, kernel_size=3,stride=1, padding=1)
        self.batch_norm70 = nn.BatchNorm2d(256, momentum=batchNorm_momentum, affine=False)
        
        self.conv82 = nn.Conv2d(256,256,kernel_size=3,stride=1,padding=1)
        self.batch_norm82 = nn.BatchNorm2d(256, momentum=batchNorm_momentum,affine=False)
        self.conv81 = nn.Conv2d(256,256, kernel_size=3, stride=1, padding=1)
        self.batch_norm81 = nn.BatchNorm2d(256, momentum=batchNorm_momentum, affine=False)
        self.conv80 = nn.Conv2d(256,128, kernel_size=3,stride=1, padding=1)
        self.batch_norm80 = nn.BatchNorm2d(128,momentum=batchNorm_momentum,affine=False)
        
        self.conv91 = nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.batch_norm91 = nn.BatchNorm2d(128, momentum=batchNorm_momentum, affine=False)
        self.conv90 = nn.Conv2d(128,64, kernel_size=3, stride=1, padding=1)
        self.batch_norm90 = nn.BatchNorm2d(64, momentum=batchNorm_momentum,affine=False)
        
        self.conv101 = nn.Conv2d(64,64, kernel_size=3,stride=1,padding=1)
        self.batch_norm101 = nn.BatchNorm2d(64, momentum=batchNorm_momentum, affine=False)
        self.conv100 = nn.Conv2d(64,n_classes,kernel_size=3, stride=1,padding=1)
        #self.batch_norm101 = nn.BatchNorm2d(n_classes,momentum=batchNorm_momentum,affine=True)
        
    def forward(self,x):
        
        #print("---- Encoder ----\n")
        x10 = F.relu(self.batch_norm10(self.conv10(x)))
        #print("x10",x10.size())
        x11 = F.relu(self.batch_norm11(self.conv11(x10)))
        #print("x11",x11.size())
        enc_1_size=x11.size()
        x1_pool,x1_ind = F.max_pool2d_with_indices(x11,kernel_size=2,stride=2,return_indices=True)
        
        x20 = F.relu(self.batch_norm20(self.conv20(x1_pool)))
        #print("x20",x20.size())
        x21 = F.relu(self.batch_norm21(self.conv21(x20)))
        x2_pool, x2_ind = F.max_pool2d_with_indices(x21,kernel_size=2,stride=2,return_indices=True)
        
        x30 = F.relu(self.batch_norm30(self.conv30(x2_pool)))
        x31 = F.relu(self.batch_norm31(self.conv31(x30)))
        x32 = F.relu(self.batch_norm32(self.conv32(x31)))
        x3_pool, x3_ind = F.max_pool2d_with_indices(x32,kernel_size=2,stride=2,return_indices=True)
        
        #print("x3_pool encoder")
        x40 = F.relu(self.batch_norm40(self.conv40(x3_pool)))
        x41 = F.relu(self.batch_norm41(self.conv41(x40)))
        x42 = F.relu(self.batch_norm42(self.conv42(x41)))
        x4_pool,x4_ind = F.max_pool2d_with_indices(x42,kernel_size=2,stride=2,return_indices=True)
        
        #print("x4_pool encoder")
        x50 = F.relu(self.batch_norm50(self.conv50(x4_pool)))
        x51 = F.relu(self.batch_norm51(self.conv51(x50)))
        x52 = F.relu(self.batch_norm52(self.conv52(x51)))
        x5_pool, x5_ind = F.max_pool2d_with_indices(x52,kernel_size=2,stride=2,return_indices=True)
        
        #print("Encoder Done \n")
        
        #print("--- Building Decoder ----\n")
        
        x5_unpool = F.max_unpool2d(x5_pool,x5_ind,kernel_size=2,stride=2)
        
        x52_dec = F.relu(self.batch_norm62(self.conv62(x5_unpool)))
        x51_dec = F.relu(self.batch_norm61(self.conv61(x52_dec)))
        x50_dec = F.relu(self.batch_norm60(self.conv60(x51_dec)))
        
        #print("x5_dec")
        x4_unpool = F.max_unpool2d(x50_dec,x4_ind,kernel_size=2,stride=2)
        
        #print("x4_unpool.size()", x4_unpool.size())
        x42_dec = F.relu(self.batch_norm72(self.conv72(x4_unpool)))
        #print("x42_dec.size()", x42_dec.size())
        x41_dec = F.relu(self.batch_norm71(self.conv71(x42_dec)))
        #print("x41_dec.size()",x41_dec.size())
        x40_dec = F.relu(self.batch_norm70(self.conv70(x41_dec)))
        
        #print("x4_dec")
        print("x40_dec size", x40_dec.shape)
        print("x3_ind", x3_ind.shape)
        x3_unpool = F.max_unpool2d(x40_dec,x3_ind,kernel_size=2,stride=2)
        
        x32_dec = F.relu(self.batch_norm82(self.conv82(x3_unpool)))
        x31_dec = F.relu(self.batch_norm81(self.conv81(x32_dec)))
        x30_dec = F.relu(self.batch_norm80(self.conv80(x31_dec)))
        #print("x3_dec")
        
        x2_unpool = F.max_unpool2d(x30_dec,x2_ind,kernel_size=2,stride=2)
        
        x21_dec = F.relu(self.batch_norm91(self.conv91(x2_unpool)))
        x20_dec = F.relu(self.batch_norm90(self.conv90(x21_dec)))
        #print("x2_dec")
        
        x1_unpool = F.max_unpool2d(x20_dec,x1_ind,kernel_size=2,stride=2)
        
        print("x1_unpool size", x1_unpool.shape)
        x11_dec = F.relu(self.batch_norm101(self.conv101(x1_unpool)))
        x10_dec = F.relu(self.conv100(x11_dec))
        
        
        x_out=F.softmax(x10_dec, dim=1)
        
        #print("----- Decoder Done ------\n")
        
        return x_out
        
        
        