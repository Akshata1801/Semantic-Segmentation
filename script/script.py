# -*- coding: utf-8 -*-
"""
Created on Mon May  4 19:31:04 2020

@author: akpo2
"""
import torch
import torchvision
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import model_segnet as segnet
#import model_segnet_skip_connections as segnet_skip
import segnet_with_skip as segnet_skip
import segnet_skip_conv as segnet_skip_conv
import torch.optim as optim
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image 
from torch.autograd import Variable
import sys
#from torchsummary import summary


EPOCHS = 100

print("--- Getting Data -----")
Data_path='D:\Deep Learning\Project\Data_DL'

transformations = transforms.Compose([
    transforms.Resize((256,512)),
    transforms.ToTensor(),   
])
train_data=datasets.Cityscapes(Data_path,split='train',mode='fine',target_type='semantic',transform=transformations,target_transform=transformations)
val_data=datasets.Cityscapes(Data_path,split='val',mode='fine',target_type='semantic',transform=transformations,target_transform=transformations)
test_data=datasets.Cityscapes(Data_path,split='test',mode='fine',target_type='semantic',transform=transformations,target_transform=transformations)

print(" ------- Data Acquired ------")

print("-------- Data Loading -------")

trainset = torch.utils.data.DataLoader(train_data, batch_size=3, shuffle=True)
valset = torch.utils.data.DataLoader(val_data, batch_size=3, shuffle=True)
testset = torch.utils.data.DataLoader(test_data, batch_size=3, shuffle=False)

print("------- Loaded Data ---------")

def main():
    if(torch.cuda.is_available()):

        device = torch.device("cuda:0")
        print("Running on GPU")
    else:
        device=torch.device("cpu")
        print("Running on cpu")
    #net=segnet_skip.segnet_skip().to(device)
    #net=segnet.segnet().to(device)
    net = segnet_skip_conv.segnet_skip_conv().to(device)
    #print(net)
    return net,device

def train(net,device):
    lr = 0.006
    weight = torch.ones(34)
    loss_function = nn.CrossEntropyLoss(weight).cuda()
    optimiser = optim.Adam(net.parameters(),lr=0.006)
    epoch_loss = []
    for epoch in range(EPOCHS):
        start_time=time.time()
        for data in trainset:
            X,y = data
            #torch.cuda.clear_memory_allocated()
            torch.cuda.empty_cache()# entirely clear all allocated memory
            X,y = X.to(device),y.to(device)
            net.zero_grad()
            output=net(X)
            output = output.view(output.size(0),output.size(1),-1)
            output = torch.transpose(output,1,2).contiguous()
            output = output.view(-1,output.size(2))
            label = y*255
            label = label.long()
            label = label.view(-1)
            loss = loss_function(output,label)
            loss.backward()
            optimiser.step()
        epoch_loss.append(loss.item())
        time_taken = time.time() - start_time


        print('=====> epoch[%d/%d] \tcur_lr: %.6f loss: %.3f time:%.2f' % (epoch + 1, EPOCHS, 0.006, loss.item(), time_taken))
    torch.save(net.state_dict(),'D:\Deep Learning\Project\Data_DL\wts_segnet.pth')
        
    
def load_pretrained_weights(net):
  vgg = torchvision.models.vgg16(pretrained=True,progress=True)
  pretrained_dict = vgg.state_dict()
  model_dict = net.state_dict()
  list1 = ['conv10.weight',
    'conv10.bias',
    'conv11.weight',
    'conv11.bias',
    'conv20.weight',
    'conv20.bias',
    'conv21.weight',
    'conv21.bias',
    'conv30.weight',
    'conv30.bias',
    'conv31.weight',
    'conv31.bias',
    'conv32.weight',
    'conv32.bias',
    'conv40.weight',
    'conv40.bias',
    'conv41.weight',
    'conv41.bias',
    'conv42.weight',
    'conv42.bias',
    'conv50.weight',
    'conv50.bias',
    'conv51.weight',
    'conv51.bias',
    'conv52.weight',
    'conv52.bias'
    ]
  list2 = ['features.0.weight',
    'features.0.bias',
    'features.2.weight',
    'features.2.bias',
    'features.5.weight',
    'features.5.bias',
    'features.7.weight',
    'features.7.bias',
    'features.10.weight',
    'features.10.bias',
    'features.12.weight',
    'features.12.bias',
    'features.14.weight',
    'features.14.bias',
    'features.17.weight',
    'features.17.bias',
    'features.19.weight',
    'features.19.bias',
    'features.21.weight',
    'features.21.bias',
    'features.24.weight',
    'features.24.bias',
    'features.26.weight',
    'features.26.bias',
    'features.28.weight',
    'features.28.bias'
    ]
  for l in range(len(list1)):
    pretrained_dict[list1[l]] = pretrained_dict.pop(list2[l])

  pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
  model_dict.update(pretrained_dict)
  net.load_state_dict(model_dict)
  return net

def decode_segmap(image, nc=34):
    
  label_colors = np.array([(0, 0, 0),  
    
               (128, 0, 0), (128,64,128), (128, 128, 0), (0, 0, 50), (128, 0, 128),

               (0, 128, 64), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),

               (198, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),

               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
               
               (0,135,0),(128,192,128),(64,128,192),(220,20,60),(64,192,128),
               
               (0, 0,190),(128,128,192),(128,192,64),(128,64,192),(192,64,128),(255,51,153),(51,51,255),(255,153,51)])
 
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
   
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
     
  rgb = np.stack([r, g, b], axis=2)
  return rgb


def test(net, device):
  correct = 0
  valset_size = len(valset)
  print("Testing")
  with torch.no_grad():
      nonzerocount = 0
      for data in valset:
          X2, y = data
          X, y = X2.to(device), y.to(device)
          output = net(X)
          for idx in range(len(output)):
            out = output[idx].cpu().max(0)[1].data.squeeze(0).byte().numpy()
            predicted_mx = output[idx]
            predicted_mx_idx = torch.argmax(predicted_mx,0)
            predicted_mx_idx = predicted_mx_idx.detach().cpu().numpy()            # finding class index with maximum softmax probability
            rgb = decode_segmap(predicted_mx_idx)
            fig = plt.figure(1)
            plt.imshow(rgb)
            plt.figure(2)
            plt.imshow(transforms.ToPILImage()(data[0][idx]))#.detach().cpu().numpy())
            plt.show()
            label = y[idx][0].detach().cpu().numpy()
            final_diff = predicted_mx_idx - label*255
            nonzerocount = nonzerocount + np.count_nonzero(final_diff)
      accu = 1 - nonzerocount/(valset_size*256*512)
      print("Accuracy",accu)

            
if __name__ == '__main__':
    sec = time.time()
    net , device = main()
    net = load_pretrained_weights(net)
    train(net,device)
    test(net,device)
    sec_last = time.time()
    print("time taken for execution",sec_last-sec)


