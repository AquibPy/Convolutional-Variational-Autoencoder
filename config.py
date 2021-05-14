import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMG_CHANNELS = 1
FEATURE_DIM = 32*20*20
ZDIM = 256
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
EPOCH = 10