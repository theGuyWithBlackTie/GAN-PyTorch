import torch
import torch.nn as nn

import config

class GeneratorNet(nn.Module):
    def __init__(self, img_shape = (config.MNIST_IMG_SIZE, config.MNIST_IMG_SIZE), latent_dim = config.LATENT_DIM):
        super(GeneratorNet, self).__init__()
        self.generated_img_shape = img_shape
        self.latent_dim          = latent_dim
        num_neurons_per_layer = [self.latent_dim, 256, 512, 1024, self.img_shape[0] * self.img_shape[1]]

        def block(in_feat, out_feat, normalize=True, activation=None):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True) if activation is None else activation)
            return layers

        # The layers are just linear layers followed by LeakyReLU and batch normalization
        # The last layer excludes batch normalization and TanH is added to map images in [-1, 1] range
        self.net = nn.Sequential(
            *block(num_neurons_per_layer[0],num_neurons_per_layer[1]),
            *block(num_neurons_per_layer[1], num_neurons_per_layer[2]),
            *block(num_neurons_per_layer[2], num_neurons_per_layer[3]),
            *block(num_neurons_per_layer[3], num_neurons_per_layer[4], normalize=False, activation=nn.Tanh())
        )


    def forward(self, latent_vector_batch):
        img_batch_flattened = self.net(latent_vector_batch)
        # unflattening using view into (N, 1, 28, 28) shape for MNIST
        return img_batch_flattened.view(img_batch_flattened.shape[0], 1, *self.generated_img_shape)


class DiscriminatorNet(nn.Module):
    def __init__(self, img_shape=(config.MNIST_IMG_SIZE, config.MNIST_IMG_SIZE)):
        super().__init__()

        num_neurons_per_layer = [img_shape[0] * img_shape[1], 512, 256, 1]

        def block(in_feat, out_feat, normalize=True, activation=None):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True) if activation is None else activation)
            return layers

        # Last layer is sigmoid function to output 1 for real images and 0 for fake images.
        self.net = nn.Sequential(
            *block(num_neurons_per_layer[0], num_neurons_per_layer[1], normalize=False),
            *block(num_neurons_per_layer[1], num_neurons_per_layer[2], normalize=False),
            *block(num_neurons_per_layer[2], num_neurons_per_layer[3], normalize=False, activation=nn.Sigmoid())
        )

    def forward(self, img_batch):
        img_batch_flattened = img_batch.view(img_batch.shape[0], -1) # flatten from (N,1,H,W) into (N, H*W)
        return self.net(img_batch_flattened)
            