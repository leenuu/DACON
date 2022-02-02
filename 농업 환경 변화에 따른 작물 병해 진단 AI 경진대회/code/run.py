import os

batch_size=16
learning_rate=0.000025
embedding_dim=512
dropout_rate=0.35
epochs=200
momentum=0.8
weigh_decay = 0.00005
img_size_x = 500
img_size_y = 500
model = 'resnet50'
optimizer = 'adamw'
scheduler = 'cosine'
rnn = "GRU"


os.system(f"python base.py {batch_size} {learning_rate} {embedding_dim} {dropout_rate} {epochs} {momentum} {img_size_x} {img_size_y} {weigh_decay}")


# os.system(f"python predict.py {batch_size} {embedding_dim} {dropout_rate} {img_size_x} {img_size_y}")

