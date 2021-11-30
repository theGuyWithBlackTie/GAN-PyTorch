import config
import engine
import model

from torchvision import transforms, datasets
from torch.optim import Adam
from torch.utils.data import DataLoader

def run():

    # transforming the images to Tensor and then bringing them in range of [-1, 1]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5, ))])

    mnist_data= datasets.MNIST(root=config.IMAGES_INPUT_PATH, train=True, download=True, transform=transform)

    train_loader = DataLoader(mnist_data, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)

    generator_net = model.GeneratorNet()
    discriminator_net = model.DiscriminatorNet()
    generator_net.to(config.DEVICE)
    discriminator_net.to(config.DEVICE)

    generator_optimizer = Adam(generator_net.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    discriminator_optimizer = Adam(discriminator_net.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))

    print('Training started')
    engine.train(generator_net, discriminator_net, generator_optimizer, discriminator_optimizer, train_loader)

    print('Training finished')
    print('Generating images')
    engine.generate_images(generator_net)


run()