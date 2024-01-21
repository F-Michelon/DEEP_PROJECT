import torch
import torch.nn as nn
import torch.optim as optim

# Define the Discriminator network for CGAN
class CGANDiscriminator(nn.Module):
    def __init__(self):
        super(CGANDiscriminator, self).__init__()
        self.label_emb = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(28*28 + 10, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Dropout(0.5),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        c = self.label_emb(labels)
        x = torch.cat([img.view(img.size(0), -1), c], 1)
        return self.model(x)
    
# Define the Generator network for CGAN
class CGANGenerator(nn.Module):
    def __init__(self):
        super(CGANGenerator, self).__init__()
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

    def forward(self, input, labels):
        c = self.label_emb(labels)
        x = torch.cat([input, c], 1)
        return self.model(x).view(-1, 1, 28, 28)
    
class CGAN(nn.Module):

    def __init__(self):
        super(CGAN, self).__init__()
        self.generator = CGANGenerator()
        self.discriminator = CGANDiscriminator()
        
    def forward(self, x, labels):
        return self.generator(x, labels)
    
    def compile(self):
        self.criterion = nn.BCELoss()
        self.optimizerG = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.optimizerD = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    def fit(self, dataloader, num_d_steps, num_epochs=20):

        history = {'d_loss': [], 'g_loss': []}

        for epoch in range(num_epochs):
            for i, (real_images, labels) in enumerate(dataloader):
                for k in range(num_d_steps):
                    # Train Discriminator
                    self.optimizerD.zero_grad()
                    real_labels = torch.ones(real_images.size(0), 1)
                    fake_labels = torch.zeros(real_images.size(0), 1)
                    outputs = self.discriminator(real_images, labels)
                    d_loss_real = self.criterion(outputs, real_labels)
                    d_loss_real.backward()

                    z = torch.randn(real_images.size(0), 100)
                    fake_images = self.generator(z, labels)
                    outputs = self.discriminator(fake_images.detach(), labels)
                    d_loss_fake = self.criterion(outputs, fake_labels)
                    d_loss_fake.backward()
                    self.optimizerD.step()

                # Train Generator
                self.optimizerG.zero_grad()
                outputs = self.discriminator(fake_images, labels)
                g_loss = self.criterion(outputs, real_labels)
                g_loss.backward()
                self.optimizerG.step()

                # Log the losses
                if (i+1) % 10 == 0:
                    print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], d_loss: {d_loss_real.item()+d_loss_fake.item()}, g_loss: {g_loss.item()}', end = '\r')
            
            # Update history
            history['d_loss'].append(d_loss_real.item()+d_loss_fake.item())
            history['g_loss'].append(g_loss.item())

            print()
        print('Finished Training')

        return history