import os

batch_size=8
learning_rate=25e-6
embedding_dim=512
dropout_rate=0.25
epochs=30
momentum=0.8
img_size_x = 640
img_size_y = 640

os.system(f"python use_numworker.py {batch_size} {learning_rate} {embedding_dim} {dropout_rate} {epochs} {momentum} {img_size_x} {img_size_y}")

# os.system(f"python predict.py {batch_size} {embedding_dim} {dropout_rate} {img_size_x} {img_size_y}")
