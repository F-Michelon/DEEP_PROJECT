import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# CGAN Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(100 + 10, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )

    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        return self.model(x).view(-1, 1, 28, 28)

# CGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(28*28 + 10, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        c = self.label_emb(labels)
        x = torch.cat([img.view(img.size(0), -1), c], 1)
        return self.model(x)

# Dataset and Dataloader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=64, shuffle=True)

# Model, Loss, Optimizer
generator = Generator()
discriminator = Discriminator()
criterion = nn.BCELoss()
optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training Loop
num_epochs = 20
for epoch in range(num_epochs):
    for i, (imgs, labels) in enumerate(dataloader):
        # Labels for generated and real data
        real = torch.ones(imgs.size(0), 1)
        fake = torch.zeros(imgs.size(0), 1)

        # Train Discriminator
        optimizerD.zero_grad()
        real_loss = criterion(discriminator(imgs, labels), real)
        real_loss.backward()

        z = torch.randn(imgs.size(0), 100)
        gen_labels = torch.randint(0, 10, (imgs.size(0),))
        fake_imgs = generator(z, gen_labels)
        fake_loss = criterion(discriminator(fake_imgs.detach(), gen_labels), fake)
        fake_loss.backward()
        optimizerD.step()

        # Train Generator
        optimizerG.zero_grad()
        g_loss = criterion(discriminator(fake_imgs, gen_labels), real)
        g_loss.backward()
        optimizerG.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {real_loss.item()+fake_loss.item()}, G Loss: {g_loss.item()}')

print("CGAN Training Complete")

import matplotlib.pyplot as plt
import numpy as np

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
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(1, num_images, i+1)
        plt.imshow(fake_images[i].squeeze(), cmap='gray')
        plt.title(f"Label: {labels[i].item()}")
        plt.axis('off')

    plt.show()

# Call the function to plot generated images
plot_generated_images_cgan(generator)
