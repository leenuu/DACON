import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# task_path = str(os.getcwd())
# add_path = task_path + "\mode\module"
# os.chdir(add_path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from glob import glob
import os
import json 
import torch
from torch import nn
from torchvision import models
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from CustomDataset import CustomDataset
import SetHelper
import CNN2RNN


train = sorted(glob('data/train/*'))
test = sorted(glob('data/test/*'))


labelsss = pd.read_csv('data/train.csv')['label']
train, val = train_test_split(train, test_size=0.2, stratify=labelsss)
train_dataset = CustomDataset(train)
val_dataset = CustomDataset(val)
test_dataset = CustomDataset(test, mode = 'test')


train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=SetHelper.batch_size, num_workers=0, shuffle=True) #16
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=SetHelper.batch_size, num_workers=0, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=SetHelper.batch_size, num_workers=0, shuffle=False)

model = CNN2RNN(max_len=SetHelper.max_len, embedding_dim=SetHelper.embedding_dim, num_features=SetHelper.num_features, class_n=SetHelper.class_n, rate=SetHelper.dropout_rate)
model = model.to(SetHelper.device)

optimizer = torch.optim.Adam(model.parameters(), lr=SetHelper.learning_rate)
criterion = nn.CrossEntropyLoss()




def accuracy_function(real, pred):    
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score

def train_step(batch_item, training):
    img = batch_item['img'].to(SetHelper.device)
    csv_feature = batch_item['csv_feature'].to(SetHelper.device)
    label = batch_item['label'].to(SetHelper.device)
    if training is True:
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            output = model(img, csv_feature)
            loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        score = accuracy_function(label, output)
        return loss, score
    else:
        model.eval()
        with torch.no_grad():
            output = model(img, csv_feature)
            loss = criterion(output, label)
        score = accuracy_function(label, output)
        return loss, score





loss_plot, val_loss_plot = [], []
metric_plot, val_metric_plot = [], []




for epoch in range(SetHelper.epochs):
    total_loss, total_val_loss = 0, 0
    total_acc, total_val_acc = 0, 0
    
    tqdm_dataset = tqdm(enumerate(train_dataloader))
    training = True
    for batch, batch_item in tqdm_dataset:
        batch_loss, batch_acc = train_step(batch_item, training)
        total_loss += batch_loss
        total_acc += batch_acc
        
        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'Loss': '{:06f}'.format(batch_loss.item()),
            'Mean Loss' : '{:06f}'.format(total_loss/(batch+1)),
            'Mean F-1' : '{:06f}'.format(total_acc/(batch+1))
        })
    loss_plot.append(total_loss/(batch+1))
    metric_plot.append(total_acc/(batch+1))
    
    tqdm_dataset = tqdm(enumerate(val_dataloader))
    training = False
    for batch, batch_item in tqdm_dataset:
        batch_loss, batch_acc = train_step(batch_item, training)
        total_val_loss += batch_loss
        total_val_acc += batch_acc
        
        tqdm_dataset.set_postfix({
            'Epoch': epoch + 1,
            'Val Loss': '{:06f}'.format(batch_loss.item()),
            'Mean Val Loss' : '{:06f}'.format(total_val_loss/(batch+1)),
            'Mean Val F-1' : '{:06f}'.format(total_val_acc/(batch+1))
        })
    val_loss_plot.append(total_val_loss/(batch+1))
    val_metric_plot.append(total_val_acc/(batch+1))
    
    if np.max(val_metric_plot) == val_metric_plot[-1]:
        torch.save(model.state_dict(), SetHelper.save_path)




def trans_data(datas):
    temp = list()
    for data in datas:
        temp.append(data.cpu().detach().numpy())
    
    return temp
loss_plot = trans_data(loss_plot)
val_loss_plot = trans_data(val_loss_plot)
plt.figure(figsize=(10,7))
plt.grid()
plt.plot(loss_plot, label='train_loss')
plt.plot(val_loss_plot, label='val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title("Loss", fontsize=25)
plt.legend()
plt.show()

plt.figure(figsize=(10,7))
plt.grid()
plt.plot(metric_plot, label='train_metric')
plt.plot(val_metric_plot, label='val_metric')
plt.xlabel('epoch')
plt.ylabel('metric')
plt.title("F-1", fontsize=25)
plt.legend()
plt.show()




def predict(dataset):
    model.eval()
    tqdm_dataset = tqdm(enumerate(dataset))
    results = []
    for batch, batch_item in tqdm_dataset:
        img = batch_item['img'].to(SetHelper.device)
        seq = batch_item['csv_feature'].to(SetHelper.device)
        with torch.no_grad():
            output = model(img, seq)
        output = torch.tensor(torch.argmax(output, dim=1), dtype=torch.int32).cpu().numpy()
        results.extend(output)
    return results

model = CNN2RNN(max_len=SetHelper.max_len, embedding_dim=SetHelper.embedding_dim, num_features=SetHelper.num_features, class_n=SetHelper.class_n, rate=SetHelper.dropout_rate)
model.load_state_dict(torch.load(SetHelper.save_path, map_location=SetHelper.device))
model.to(SetHelper.device)

preds = predict(test_dataloader)
preds = np.array([CustomDataset.label_decoder[int(val)] for val in preds])




submission = pd.read_csv('data/sample_submission.csv')
submission['label'] = preds
submission.to_csv('baseline_submission.csv', index=False)