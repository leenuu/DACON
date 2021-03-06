from multiprocessing.dummy import freeze_support
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import sys
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
from torchvision import transforms
import torchvision
from torch.utils.data import Dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import timm
from PIL import Image
from keras import metrics

# from tensorflow.keras import layers
# import tensorflow as tf




class CustomDataset(Dataset):
    def __init__(self, files, csv_feature_dict, label_encoder, labels=None, mode='train'):
        self.mode = mode
        self.files = files
        self.csv_feature_dict = csv_feature_dict
        self.csv_feature_check = [0]*len(self.files)
        self.csv_features = [None]*len(self.files)
        self.max_len = 24 * 6
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, i):
        file = self.files[i]
        file_name = file.split('\\')[-1]
        
        # csv
        if self.csv_feature_check[i] == 0:
            csv_path = f'{file}/{file_name}.csv'
            df = pd.read_csv(csv_path)[self.csv_feature_dict.keys()]
            df = df.replace('-', 0)
            # MinMax scaling
            for col in df.columns:
                df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
                df[col] = df[col] / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])
            # zero padding
            pad = np.zeros((self.max_len, len(df.columns)))
            length = min(self.max_len, len(df))
            pad[-length:] = df.to_numpy()[-length:]
            # transpose to sequential data
            csv_feature = pad.T
            self.csv_features[i] = csv_feature
            self.csv_feature_check[i] = 1
        else:
            csv_feature = self.csv_features[i]
        

        # image
        image_path = f'{file}/{file_name}.jpg'
        img = cv2.imread(image_path)
        PIL_image = Image.fromarray(img)


        transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize((int(sys.argv[7]), int(sys.argv[8]))),
            torchvision.transforms.RandomHorizontalFlip(p = 1),
            torchvision.transforms.RandomVerticalFlip(p = 1),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        # img = cv2.resize(img, dsize=(int(sys.argv[7]), int(sys.argv[8])), interpolation=cv2.INTER_AREA)

        
        img = transforms(PIL_image)

        # img = img.astype(np.float32)/255
        # img = np.transpose(img, (2,0,1))
        

        

        if self.mode == 'train':
            json_path = f'{file}/{file_name}.json'
            with open(json_path, 'r') as f:
                json_file = json.load(f)
            
            crop = json_file['annotations']['crop']
            disease = json_file['annotations']['disease']
            risk = json_file['annotations']['risk']
            label = f'{crop}_{disease}_{risk}'
            
            return {
                # 'img' : torch.tensor(img, dtype=torch.float32),
                'img' : img,
                'csv_feature' : torch.tensor(csv_feature, dtype=torch.float32),
                'label' : torch.tensor(self.label_encoder[label], dtype=torch.long)
            }
        else:
            return {
                # 'img' : torch.tensor(img, dtype=torch.float32),
                'img' : img,
                'csv_feature' : torch.tensor(csv_feature, dtype=torch.float32)
            }




class CNN_Encoder(nn.Module):
    def __init__(self, class_n, rate=0.1):
        super(CNN_Encoder, self).__init__()
        self.model = models.resnet50(pretrained=True)
        # self.model = timm.create_model('tf_efficientnet_b6_ns', pretrained=True)#, num_classes=class_n)
        # self.model = timm.create_model('regnetx_006', pretrained=True)
        # self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=True)
        self.model = self.model.float()

    
    def forward(self, inputs):
        output = self.model(inputs)
        return output





class RNN_Decoder(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, class_n, rate):
        super(RNN_Decoder, self).__init__()
        # self.lstm = nn.LSTM(max_len, embedding_dim)
        self.lstm = nn.GRU(max_len, embedding_dim)
        self.rnn_fc = nn.Linear(num_features*embedding_dim, 1000)
        self.final_layer = nn.Linear(1000 + 1000, class_n) # resnet out_dim + lstm out_dim
        self.dropout = nn.Dropout(rate)

    def forward(self, enc_out, dec_inp):
        hidden, _ = self.lstm(dec_inp)
        hidden = hidden.view(hidden.size(0), -1)
        hidden = self.rnn_fc(hidden)
        concat = torch.cat([enc_out, hidden], dim=1) # enc_out + hidden 
        fc_input = concat
        output = self.dropout((self.final_layer(fc_input)))
        return output






class CNN2RNN(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, class_n, rate):
        super(CNN2RNN, self).__init__()
        self.cnn = CNN_Encoder(embedding_dim, rate)
        self.rnn = RNN_Decoder(max_len, embedding_dim, num_features, class_n, rate)
        
    def forward(self, img, seq):
        cnn_output = self.cnn(img)
        output = self.rnn(cnn_output, seq)
        
        return output





def run():
    freeze_support()

    # ????????? ????????? feature ??????
    csv_features = ['?????? ?????? 1 ??????', '?????? ?????? 1 ??????', '?????? ?????? 1 ??????', '?????? ?????? 1 ??????', '?????? ?????? 1 ??????', 
                    '?????? ?????? 1 ??????', '?????? ????????? ??????', '?????? ????????? ??????', '?????? ????????? ??????']

    csv_files = sorted(glob('data/train/*/*.csv'))

    temp_csv = pd.read_csv(csv_files[0])[csv_features]
    max_arr, min_arr = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()

    # feature ??? ?????????, ????????? ??????
    for csv in tqdm(csv_files[1:]):
        temp_csv = pd.read_csv(csv)[csv_features]
        temp_csv = temp_csv.replace('-',np.nan).dropna()
        if len(temp_csv) == 0:
            continue
        temp_csv = temp_csv.astype(float)
        temp_max, temp_min = temp_csv.max().to_numpy(), temp_csv.min().to_numpy()
        max_arr = np.max([max_arr,temp_max], axis=0)
        min_arr = np.min([min_arr,temp_min], axis=0)

    # feature ??? ?????????, ????????? dictionary ??????
    csv_feature_dict = {csv_features[i]:[min_arr[i], max_arr[i]] for i in range(len(csv_features))}




    # ?????? ?????? csv ?????? ??????
    crop = {'1':'??????','2':'?????????','3':'????????????','4':'??????','5':'??????','6':'????????????'}
    disease = {'1':{'a1':'????????????????????????','a2':'??????????????????','b1':'????????????','b6':'?????????????????? (N)','b7':'?????????????????? (P)','b8':'?????????????????? (K)'},
            '2':{'a5':'?????????????????????','a6':'???????????????????????????','b2':'??????','b3':'????????????','b6':'?????????????????? (N)','b7':'?????????????????? (P)','b8':'?????????????????? (K)'},
            '3':{'a9':'????????????????????????','a10':'?????????????????????','b3':'????????????','b6':'?????????????????? (N)','b7':'?????????????????? (P)','b8':'?????????????????? (K)'},
            '4':{'a3':'???????????????','a4':'??????????????????','b1':'????????????','b6':'?????????????????? (N)','b7':'?????????????????? (P)','b8':'?????????????????? (K)'},
            '5':{'a7':'???????????????','a8':'??????????????????','b3':'????????????','b6':'?????????????????? (N)','b7':'?????????????????? (P)','b8':'?????????????????? (K)'},
            '6':{'a11':'?????????????????????','a12':'?????????????????????','b4':'????????????','b5':'?????????'}}
    risk = {'1':'??????','2':'??????','3':'??????'}
    label_description = {}
    for key, value in disease.items():
        label_description[f'{key}_00_0'] = f'{crop[key]}_??????'
        for disease_code in value:
            for risk_code in risk:
                label = f'{key}_{disease_code}_{risk_code}'
                label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'



    label_encoder = {key:idx for idx, key in enumerate(label_description)}
    label_decoder = {val:key for key, val in label_encoder.items()}




    batch_size = int(sys.argv[1])
    learning_rate = float(sys.argv[2])
    embedding_dim = int(sys.argv[3])
    dropout_rate = float(sys.argv[4])
    epochs = int(sys.argv[5])
    momentums = float(sys.argv[6])
    # save_path = str(sys.argv[9])


    device = torch.device("cuda:0")
    class_n = len(label_encoder)
    num_features = len(csv_feature_dict)
    max_len = 24*6
    vision_pretrain = True



    train = sorted(glob('data/train/*'))
    test = sorted(glob('data/test/*'))

    labelsss = pd.read_csv('data/train.csv')['label']
    train, val = train_test_split(train, test_size=0.2, stratify=labelsss)
    train_dataset = CustomDataset(train, csv_feature_dict=csv_feature_dict, label_encoder=label_encoder)
    val_dataset = CustomDataset(val, csv_feature_dict=csv_feature_dict, label_encoder=label_encoder)
    test_dataset = CustomDataset(test, csv_feature_dict=csv_feature_dict, label_encoder=label_encoder, mode = 'test')

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True) #16
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=False)


    model = CNN2RNN(max_len=max_len, embedding_dim=embedding_dim, num_features=num_features, class_n=class_n, rate=dropout_rate)
    model = model.to(device)




    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=float(sys.argv[9]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=float(sys.argv[9]))
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentums, weight_decay=float(sys.argv[9]))
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=float(sys.argv[9]))
   
   
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=1, patience=1, mode='min')

    criterion = nn.CrossEntropyLoss()


    def accuracy_function(real, pred):    
        real = real.cpu()
        pred = torch.argmax(pred, dim=1).cpu()
        score = f1_score(real, pred, average='macro')
        return score




    def train_step(batch_item, training):
        img = batch_item['img'].to(device)
        csv_feature = batch_item['csv_feature'].to(device)
        label = batch_item['label'].to(device)
        if training is True:
            model.train()
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                output = model(img, csv_feature)
                loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
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






    for epoch in range(epochs):
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
        
        torch.save(model.state_dict(), f'Epoch {epoch + 1}.pt')

        # if np.max(val_metric_plot) == val_metric_plot[-1]:
        #     torch.save(model.state_dict(), save_path)





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

    print(f'MAX F-1 : {max(val_metric_plot)}')





    def predict(dataset):
        model.eval()
        tqdm_dataset = tqdm(enumerate(dataset))
        results = []
        for batch, batch_item in tqdm_dataset:
            img = batch_item['img'].to(device)
            seq = batch_item['csv_feature'].to(device)
            with torch.no_grad():
                output = model(img, seq)
            output = torch.tensor(torch.argmax(output, dim=1), dtype=torch.int32).cpu().numpy()
            results.extend(output)
        return results
    model_number = input("model number : ")
    model = CNN2RNN(max_len=max_len, embedding_dim=embedding_dim, num_features=num_features, class_n=class_n, rate=dropout_rate)
    model.load_state_dict(torch.load(f'Epoch {model_number}.pt', map_location=device))
    model.to(device)

    preds = predict(test_dataloader)
    preds = np.array([label_decoder[int(val)] for val in preds])



    submission = pd.read_csv('data/sample_submission.csv')
    submission['label'] = preds
    submission.to_csv(f'Epoch {model_number}.csv', index=False)




if __name__ == '__main__':
    run()
