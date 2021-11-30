import time

import torch
import torch.nn as nn
from torchvision.utils import save_image

import config

def get_gaussian_latent_batch(batch_size, latent_size, device = config.DEVICE):
    return torch.randn((batch_size, latent_size), device = device)



def train(generator, discriminator, optimizer_G, optimizer_D, train_loader, num_epochs = config.EPOCHS, device = config.DEVICE):
    discriminator_losses = []
    generator_losses     = []

    discriminator.train()
    generator.train()

    # Using BCE loss for training Generator and Discriminator
    adversarial_loss = torch.nn.BCELoss()
    # Ground truth for real images is 1
    real_images_ground_truth = torch.ones((config.BATCH_SIZE, 1), device = device)
    # Ground truth for fake images is 0
    fake_images_ground_truth = torch.zeros((config.BATCH_SIZE, 1), device = device)


    # constants for tracking training
    console_log_freq   = 50
    debug_imagery_freq = 50
    checkpoint_freq    = 2
    ts                 = time.time()

    for epoch in range(num_epochs):
        for batch_idx, (real_images, _) in enumerate(train_loader):
            real_images = real_images.to(device)

            #
            # Train Discriminator
            #
            optimizer_D.zero_grad()
            
            real_images_prediction = discriminator(real_images)
            real_images_loss = adversarial_loss(real_images_prediction, real_images_ground_truth)
            
            # Generate fake images
            latent_batch           = get_gaussian_latent_batch(config.BATCH_SIZE, config.LATENT_DIM)
            fake_images            = generator(latent_batch)
            # Calling detatch() to detach the fake images from the computation graph for Discriminator
            fake_images_prediction = discriminator(fake_images.detach())
            fake_images_loss       = adversarial_loss(fake_images_prediction, fake_images_ground_truth)

            discriminator_loss = real_images_loss + fake_images_loss
            discriminator_loss.backward()
            optimizer_D.step()


            #
            # Train Generator
            #

            optimizer_G.zero_grad()
            # Generate fake images
            latent_batch             = get_gaussian_latent_batch(config.BATCH_SIZE, config.LATENT_DIM)
            fake_images              = generator(latent_batch)
            discriminator_prediction = discriminator(fake_images)
            generator_loss           = adversarial_loss(discriminator_prediction, real_images_ground_truth)
            generator_loss.backward()
            optimizer_G.step()

            if batch_idx % console_log_freq == 0:
                prefix = 'GAN training: time elapsed'
                generator_losses.append(generator_loss.item())
                discriminator_losses.append(discriminator_loss.item())
                print(f'{prefix} = {(time.time() - ts):.2f} [s] | epoch={epoch + 1} | batch= [{batch_idx + 1}/{len(train_loader)}] | G_Loss: [{generator_loss.item()}] | D_Loss: [{discriminator_loss.item()}]')


            # Saving intermediary generator images
            if batch_idx % debug_imagery_freq == 0:
                with torch.no_grad():
                    latent_batch = get_gaussian_latent_batch(config.BATCH_SIZE, config.LATENT_DIM)
                    fake_images = generator(latent_batch)
                    fake_images = fake_images.detach().cpu()

                    fake_images_resized = nn.Upsample(scale_factor = 2.5, mode = 'nearest')(fake_images)
                    save_image(fake_images_resized, f'{config.IMAGES_OUTPUT_PATH}/{epoch}_{batch_idx}.png')


            # Save generator checkpoint
        if (epoch + 1) % checkpoint_freq == 0:
            torch.save(generator.state_dict(), f'{config.MODEL_OUTPUT_PATH}/generator_checkpoint_{epoch}.pt')

    # Saving final generator checkpoint
    print('Saving final checkpoint...')
    torch.save(generator.state_dict(), f'{config.MODEL_OUTPUT_PATH}/generator_checkpoint_final.pt')




def generate_images(generator, device = config.DEVICE):
    with torch.no_grad():
        latent_batch = get_gaussian_latent_batch(config.BATCH_SIZE, config.LATENT_DIM)
        fake_images = generator(latent_batch)
        fake_images = fake_images.detach().cpu()

        fake_images_resized = nn.Upsample(scale_factor = 2.5, mode = 'nearest')(fake_images)
        save_image(fake_images_resized, f'{config.IMAGES_OUTPUT_PATH}/final.png')
                