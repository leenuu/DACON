import torch
import CustomDataset

class SetHelper():
    def __init__(self):
        pm = self.set_Parameters()
        self.batch_size = int(pm["batch_size"])
        self.learning_rate = float(pm["learning_rate"])
        self.embedding_dim = int(pm["embedding_dim"])
        self.dropout_rate = float(pm["dropout_rate"])
        self.epochs = int(pm["epochs"])
        self.save_path = pm["save_path"]


        self.device = torch.device("cuda:0")
        self.class_n = len(CustomDataset.label_encoder)
        self.num_features = len(CustomDataset.csv_feature_dict)
        self.max_len = 24*6
        self.vision_pretrain = True

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