import imp
from torch import nn
from torchvision import models
import torch
import CNN_Encoder
import RNN_Decoder
import CustomDataset


max_len = 24*6

class CNN2RNN(nn.Module):
    def __init__(self, max_len, embedding_dim, num_features, class_n, rate):
        super(CNN2RNN, self).__init__()
        self.cnn = CNN_Encoder(embedding_dim, rate)
        self.rnn = RNN_Decoder(max_len, embedding_dim, num_features, class_n, rate)
        pm = self.set_Parameters()
        self.batch_size = int(pm["batch_size"])
        self.learning_rate = float(pm["learning_rate"])
        self.embedding_dim = int(pm["embedding_dim"])
        self.dropout_rate = float(pm["dropout_rate"])
        self.epochs = int(pm["epochs"])
        self.save_path = pm["save_path"]


        self.device = torch.device("cuda:0")
        class_n = len(CustomDataset.label_encoder)
        num_features = len(CustomDataset.csv_feature_dict)
        max_len = 24*6
        self.vision_pretrain = True
        
    def forward(self, img, seq):
        cnn_output = self.cnn(img)
        output = self.rnn(cnn_output, seq)
        
        return output
    
    def set_Parameters():
        with open('Parameter.txt') as lns:
            Parameters = dict()
            data = lns.read()
            Parameter = data.replace("\n", "").split(",")
            for temp in Parameter:
                p_name = temp.split("=")[0]
                P_val = temp.split("=")[1]
                Parameters[p_name] = P_val

        return Parameters