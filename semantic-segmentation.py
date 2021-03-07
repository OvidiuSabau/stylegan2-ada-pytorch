#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torch.autograd import Variable
import sys
import pickle
import matplotlib.pyplot as plt
if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
else:
    device = torch.device("cpu")
    print("Running on the CPU")
import itertools


# In[2]:


# all_files = []
# func = np.vectorize(lambda x: x!=0 and x!=1 and x!=2)

# for idx, filename in enumerate(os.listdir('FullyProccessedCelebA/')):
    
#     if idx % 500 == 0:
#         print(idx)
        
#     arr = np.load('FullyProccessedCelebA/'+filename)
#     newArr = np.empty((256, 256, 4), dtype=np.uint8)
#     newArr[:, :, :3] = arr[:, :, :3]
    
#     argmax = np.argmax(arr[:, :, 3:], axis=-1)
#     newArr[:, :, 3] = argmax
#     newArr = np.moveaxis(newArr, -1, 0)
    
#     all_files.append(newArr.reshape(4, 256, 256))
    
#     if func(newArr[3]).any():
#         print(argmax)


# In[3]:


# all_files = np.stack(all_files)


# In[4]:


# test_size = all_files.shape[0] // 10
# np.save('preprocessed-datasets/celebmask-test.npy', all_files[:test_size])
# np.save('preprocessed-datasets/celebmask-dev.npy', all_files[test_size:test_size * 2])
# np.save('preprocessed-datasets/celebmask-train.npy', all_files[test_size * 2:])


# In[5]:


class CNN(nn.Module):
    def __init__(
        self,
        in_channels,
        expansion_rate,
        num_layers,
        kernel_size,
        segmentation_channels
    ):
        super(CNN, self).__init__()
        self.in_channels = in_channels
        self.expansion_rate = expansion_rate
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.segmentation_channels = segmentation_channels
        
        self.layers = nn.ModuleDict()
        
        current_channels = in_channels
        
        self.layers['first_bn'] = nn.BatchNorm2d(num_features=in_channels)
        
        for layer in range(num_layers):
            
            if current_channels > 4 * expansion_rate:
                
                self.layers['bottleneck' + str(layer)] = nn.Conv2d(in_channels=current_channels, 
                                    out_channels=4 * expansion_rate, kernel_size=1, padding=0)
                
                self.layers['conv' + str(layer)] = nn.Conv2d(
                    in_channels=4 * expansion_rate, out_channels=expansion_rate,
                    kernel_size=kernel_size, padding=(kernel_size-1)//2)
                
            else:
                self.layers['conv' + str(layer)] = nn.Conv2d(
                    in_channels=current_channels, out_channels=expansion_rate,
                    kernel_size=kernel_size, padding=(kernel_size-1)//2)

            
            self.layers['bn' + str(layer)] = nn.BatchNorm2d(num_features=expansion_rate)
            
            current_channels += expansion_rate
            
        self.layers['final'] = nn.Conv2d(
                in_channels=current_channels, out_channels=segmentation_channels, 
                kernel_size=kernel_size, padding=(kernel_size-1)//2)
                
    def forward(self, x):
         
        out = x
        
        out = self.layers['first_bn'](out)
        
        for layer in range(self.num_layers):
            
            if 'bottleneck' + str(layer) in self.layers.keys():
                tmp_out = self.layers['bottleneck' + str(layer)](out)
                tmp_out = self.layers['conv' + str(layer)](tmp_out)
            else:
                tmp_out = self.layers['conv' + str(layer)](out)
            tmp_out = self.layers['bn' + str(layer)](tmp_out)
            tmp_out = torch.relu(tmp_out)
            
            out = torch.cat((out, tmp_out), dim=1)
        
        out = self.layers['final'](out)
        
        return out


# In[6]:


trainingData = np.load('preprocessed-datasets/celebmask-train.npy')
testingData = np.load('preprocessed-datasets/celebmask-test.npy')


# In[31]:


criterion = nn.CrossEntropyLoss()
train_batch_size = 32
test_batch_size = 32
num_train_batches = int(np.ceil(trainingData.shape[0] / train_batch_size))
num_test_batches = int(np.ceil(testingData.shape[0] / test_batch_size))
expansion_rate = 12
num_layers = 3
in_channels = 3
segmentation_channels = 3
kernel_size = 3
lr = 1e-3
model = CNN(in_channels=in_channels, expansion_rate=expansion_rate, num_layers=num_layers, kernel_size=kernel_size, segmentation_channels=segmentation_channels)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)
lr_lambda = lambda epoch: 0.6
lr_scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)

print('Num training batches', num_train_batches)
print('Num testing batches', num_test_batches)


# In[32]:


def print_stats(batch, trainLoss, acc, parameters, time):
    
    s = 0
    for parameter in parameters:
        s += torch.abs(parameter.grad).mean().detach().cpu()
    
    print('Batch {} in {:.1f}s | Loss {:.4f} | Acc {:.3f} | Grad {:.1f}'.format(
        batch, time, trainLoss, acc, np.log10(s)))


# In[33]:


testLosses = []
testAcc = []
trainLosses = []
trainAcc = []


# In[ ]:


for epoch in range(5):
    
    print('Starting Epoch {}'.format(epoch))
    epoch_t0 = time.time()
        
    testLosses.append(0)
    testAcc.append(0)
    
    model.eval()
    with torch.no_grad():
        
        for batch in range(num_test_batches):
            
            # 
            x = torch.from_numpy(testingData[batch * test_batch_size : (batch + 1) * test_batch_size, :3]).float().to(device)
            y = torch.from_numpy(testingData[batch * test_batch_size : (batch + 1) * test_batch_size, 3]).long().to(device)
            
            y_hat = model(x)
            loss = criterion(y_hat, y)
            argmax = torch.argmax(y_hat, dim=1)
            acc = (torch.sum(torch.eq(argmax, y)) / y.nelement()).cpu().detach()
            
            testLosses[-1] += loss.item()
            testAcc[-1] += acc
        
        testLosses[-1] /= num_test_batches
        testAcc[-1] /= num_test_batches

        print('Test Loss {:.4f} Acc {:.4f}'.format(testLosses[-1], testAcc[-1]))
        
    
    permutation = np.random.permutation(trainingData.shape[0])
    trainingData = trainingData[permutation]
    
    model.train()
    for batch in range(num_train_batches):
                
        t0 = time.time()
        
        x = torch.from_numpy(trainingData[batch * train_batch_size : (batch + 1) * train_batch_size, :3]).float().to(device)
        y = torch.from_numpy(trainingData[batch * train_batch_size : (batch + 1) * train_batch_size, 3]).long().to(device)

        y_hat = model(x)
        loss = criterion(y_hat, y)

        argmax = torch.argmax(y_hat, dim=1)
        acc = (torch.sum(torch.eq(argmax, y)) / y.nelement()).cpu().detach()

        trainLosses.append(loss.item())
        trainAcc.append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        t_final = time.time() - t0
        
        if batch % 50 == 0:
            lr_scheduler.step()
            print_stats(batch, loss.item(), acc, model.parameters(), t_final)
    
    print('Epoch {} finished in {:.1f} w/ lr {}'.format(epoch, time.time() - epoch_t0, lr_scheduler.get_last_lr()[0]))


# In[30]:


torch.save(model, 'semantic-segmentation.pt')

