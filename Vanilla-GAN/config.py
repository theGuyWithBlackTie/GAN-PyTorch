import torch

import os

MNIST_IMG_SIZE = 28
LATENT_DIM     = 100
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS         = 10
BATCH_SIZE     = 128
LEARNING_RATE  = 0.0002


IMAGES_INPUT_PATH  = os.path.join(os.getcwd(), '..', 'data')
IMAGES_OUTPUT_PATH =  os.path.join(os.getcwd() ,'debug_imagery')
MODEL_OUTPUT_PATH  =  os.path.join(os.getcwd() , 'models')