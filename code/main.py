from cgan import CGAN
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from pathlib import Path

if __name__ == '__main__':

    # Load MNIST dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    batch_size = 64
    dataloader = DataLoader(mnist, batch_size=batch_size, shuffle=True)

    # Initialize models
    cgan = CGAN()
    cgan.compile()

    # Training loop
    num_epochs = 30
    num_d_steps = 1 # Number of discriminator steps per generator step
    history = cgan.fit(dataloader, num_d_steps=num_d_steps, num_epochs=num_epochs)

    # Save the model
    filename = 'baseline_cgan_more_dropout_k3'
    torch.save(cgan, str(Path('saved_models/{}.pth'.format(filename)).absolute()))

    # Save history figure
    fig, ax = plt.subplots()
    ax.plot(history['d_loss'], label='Discriminator Loss')
    ax.plot(history['g_loss'], label='Generator Loss')
    ax.legend()
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    fig.savefig('figures/{}.png'.format(filename))

    # Function to generate and plot images from the CGAN generator
    def plot_generated_images_cgan(generator, num_images=10):
        # Generate noise vectors
        z = torch.randn(num_images, 100)

        # Specify labels for each image (e.g., digits 0-9)
        labels = torch.tensor([i for i in range(num_images)])

        # Generate images from the noise vectors and specified labels
        with torch.no_grad():  # We don't need to track gradients here
            fake_images = generator(z, labels).cpu()

        # Plot the images
        fig, ax = plt.subplots(2,5)
        for i in range(num_images):
            ax[i//5,i%5].imshow(fake_images[i].squeeze(), cmap='gray')
            ax[i//5,i%5].set_title(f"Label: {labels[i].item()}")
        fig.savefig('figures/images_{}.png'.format(filename))

    # Call the function to plot generated images
    plot_generated_images_cgan(cgan)