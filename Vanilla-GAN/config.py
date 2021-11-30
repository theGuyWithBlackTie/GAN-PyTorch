import torch

MNIST_IMG_SIZE = 28
LATENT_DIM     = 100
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS         = 10