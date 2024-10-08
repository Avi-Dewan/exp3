import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, classes=10, latent_dim=100, img_size=32, channels=3):
        super().__init__()

        self.label_emb = nn.Embedding(classes, classes)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim + classes, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and noise to form input
        gen_input = torch.cat((noise, self.label_emb(labels)), -1)
        out = self.l1(gen_input)
        out = out.view(out.size(0), 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, classes=10, img_size=32, channels=3):
        super().__init__()

        self.label_embedding = nn.Embedding(classes, classes)
        self.img_size = img_size
        self.classes = classes

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels + classes, 64, bn=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(512 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img, labels):
        # Concatenate image and label condition
        labels = self.label_embedding(labels)
        labels = labels.unsqueeze(-1).unsqueeze(-1).expand(img.size(0), self.classes, self.img_size, self.img_size)
        d_in = torch.cat((img, labels), 1)
        out = self.conv_blocks(d_in)
        out = out.view(out.size(0), -1)
        validity = self.adv_layer(out)
        return validity
