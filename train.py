import torch
import numpy as np
import pickle
import json
import csv
import imageio
import os
import pandas as pd
from torchvision import datasets, models, transforms
from PIL import Image
from torch import nn
import scipy
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
from torch.utils.data import Dataset, DataLoader
import time
import random
from torch.optim.lr_scheduler import StepLR
import shutil
from datetime import date
import scipy.io as sio


raw_features = sio.loadmat('./wikipedia_info/raw_features.mat')

img_features, text_features = np.float32(raw_features['I_tr']), np.float32(raw_features['T_tr'])

concepts = []

infile = open('./wikipedia_info/trainset_txt_img_cat.list', 'r')
data = infile.readlines()
for line in data:
    if(int(line[-2])==0):
        concepts.append(9)
    else:
        concepts.append(int(line[-2])-1)


#Make all text and img pairs
#Store img index,text index,img features,text features,img label,text label,label_similarity

pairs = []

for i in range(len(img_features)):
    arr = np.arange(len(img_features))
    np.random.shuffle(arr)
    
    for j in arr[:100]:
        pairs.append((i, j, img_features[i], text_features[j], concepts[i], concepts[j], np.float32(concepts[i]==concepts[j])))

pairs.append((i, i, np.float32(img_features[i]) , np.float32(text_features[i]), concepts[i], concepts[i], 1.))


class DataLoader():
    """
    DataLoader class for image features of the WIKI dataset.
    """
    
    def __init__(self, img_text_pairs, train=True):
        #If train is false, we are testing
        self.all_pairs = img_text_pairs
        
        self.num_img_features = len(img_text_pairs[0][2])
        self.num_text_features = len(img_text_pairs[0][3])
         
    def __len__(self):
        return len(self.all_pairs)
    
    def __getitem__(self, idx):
        pair_info = self.all_pairs[idx]
        return [{'img' : pair_info[0], 'text': pair_info[1], 'img_feats': pair_info[2],
                 'text_feats': pair_info[3], 'img_label': pair_info[4], 'text_label':pair_info[5],
                 'similarity': pair_info[6]}]


class Model(nn.Module):
    
    def __init__(self, hashing_length=32, num_of_concepts=10):
        super(Model, self).__init__()
        
        self.img_hashing_layer = nn.Linear(128, hashing_length)
        nn.init.xavier_normal_(self.img_hashing_layer.weight)
        
        self.text_hashing_layer = nn.Linear(10, hashing_length)
        nn.init.xavier_normal_(self.text_hashing_layer.weight)
        
        self.img_output = nn.Linear(hashing_length, num_of_concepts)
        nn.init.xavier_normal_(self.img_output.weight)
        
        self.text_output = nn.Linear(hashing_length, num_of_concepts)
        nn.init.xavier_normal_(self.text_output.weight)
        
        self.num_of_concepts = num_of_concepts
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        
    def hash_vector(self, features, img=True):
        if img:
            hash_code = self.img_hashing_layer(features)
        else:
            hash_code = self.text_hashing_layer(features)
            
        hash_code = self.tanh(hash_code)
        return hash_code
        
    def forward(self, img_features, text_features):
        img_hash_code = self.img_hashing_layer(img_features)
        img_hash_code = self.tanh(img_hash_code)
        concept_vector = self.img_output(img_hash_code)
        img_output = self.softmax(concept_vector)

        text_hash_code = self.text_hashing_layer(text_features)
        text_hash_code = self.tanh(text_hash_code)
        concept_vector = self.text_output(text_hash_code)
        text_output = self.softmax(concept_vector)
        
        return img_output, text_output, img_hash_code, text_hash_code


class Loss_Correction(nn.Module):
    def __init__(self,):
        super(Loss_Correction, self).__init__()

    def forward(self, img_hash_batch, text_hash_batch):
        #Quantization
        loss = torch.norm(torch.abs(img_hash_batch) - torch.ones(img_hash_batch.shape), p='fro')**2
        loss += torch.norm(torch.abs(text_hash_batch) - torch.ones(text_hash_batch.shape), p='fro')**2

        #Bit Balancing
        loss += torch.norm(torch.matmul(torch.ones((1,img_hash_batch.shape[0])), img_hash_batch), p=2)**2
        loss += torch.norm(torch.matmul(torch.ones((1,img_hash_batch.shape[0])), text_hash_batch), p=2).item()**2

        return loss/2*(img_hash_batch.shape[0])


def transform_to_binary(hash_code):
    binary_code = np.ones(len(hash_code))
    for i, el in enumerate(hash_code):
        if (el<0):
            binary_code[i] = -1
    
    return binary_code


def loss_fn_1(img_hash_code, text_hash_code, similarity):    
    #Inter Modality Similarity preserving loss
    loss1 = nn.MSELoss()
    return loss1(torch.dot(img_hash_code, text_hash_code)/len(img_hash_code), similarity)


def loss_fn_2(img_output, text_output, img_label, text_label): 
    #Label Preserving loss
    loss2 = nn.CrossEntropyLoss()
    return (loss2(img_output, img_label) + loss2(text_output, text_label))


BATCH_SIZE = 64
NUM_OF_EPOCHS = 1000


loss1 = nn.MSELoss()
loss1(torch.zeros((1,)), torch.ones((1,))).item()

train_dataset = DataLoader(pairs)

train_size = int(0.9 * len(train_dataset))
valid_size = len(train_dataset) - train_size

train_ds, valid_ds = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

train_loader = torch.utils.data.DataLoader(train_ds,batch_size=BATCH_SIZE, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_ds,batch_size=BATCH_SIZE, shuffle=True)


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

model = Model(hashing_length=32, num_of_concepts=10)
model = model.to(device)

model.train()

optimizer = optim.Adagrad(
    [
        {"params": model.img_hashing_layer.parameters(), "lr": 0.01},
        {"params": model.text_hashing_layer.parameters(), "lr": 0.01},
        {"params": model.img_output.parameters(), "lr": 0.001},
        {"params": model.text_output.parameters(), "lr": 0.001},
    ],
    lr=0.00001,
    )

alpha, beta = 1, 0.5

for epoch in range(NUM_OF_EPOCHS):
    total_epoch_loss = 0
    optimizer.step()

    for i, data_row in enumerate(train_loader):
        optimizer.zero_grad()

        batch_img_feats, batch_text_feats, batch_img_label, batch_text_label, batch_similarity = data_row[0]['img_feats'].to(device), data_row[0]['text_feats'].to(device),\
                                                                                                 data_row[0]['img_label'].to(device), data_row[0]['text_label'].to(device),\
                                                                                                 data_row[0]['similarity'].to(device)

        img_output, text_output, img_hash_batch, text_hash_batch = model(batch_img_feats.float(), batch_text_feats.float()) 

        for j in range(img_output.shape[0]):
            if (j==0):
                loss1 = loss_fn_1(img_hash_batch[j], text_hash_batch[j], batch_similarity[j])
            else:
                loss1 += loss_fn_1(img_hash_batch[j], text_hash_batch[j], batch_similarity[j]) 

        loss2 = loss_fn_2(img_output, text_output, batch_img_label, batch_text_label)

        correction = Loss_Correction()
        loss3 = correction(img_hash_batch, text_hash_batch)

        loss = alpha*(loss1+loss2)
#         loss = alpha*(loss1+loss2) + beta*loss3
        loss.backward()

        optimizer.step()
        
        total_epoch_loss += loss.item()

    print("Epoch %d, Average loss per batch : %0.3f"%(epoch, total_epoch_loss/len(train_loader)))
    
    if(epoch%10==0):
        torch.save(model.state_dict(), "./saved_models/epoch" + str(epoch) + ".pth")

