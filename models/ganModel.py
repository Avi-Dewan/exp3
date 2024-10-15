import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, classes=10, latent_dim=100, img_size=32, channels=3):
        super(Generator, self).__init__()

        self.img_size = img_size
        self.channels = channels

        # Label embedding to project class labels into latent space
        self.label_embedding = nn.Embedding(classes, 50)

        # Fully connected layer for latent vector and label
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim + 50, 128 * 8 * 8),  # (latent_dim + label embedding size) -> 8x8x128
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Transpose convolutions to upsample the image from 8x8 to 32x32
        self.conv_blocks = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
            nn.LeakyReLU(0.2, inplace=True),
            # Adjusting final layer to ensure output is 32x32
            nn.Conv2d(128, channels, kernel_size=3, padding=1),  # Output is [channels, img_size, img_size]
            nn.Tanh()  # Output normalized between [-1, 1]
        )

    def forward(self, noise, labels):
        # Embed the labels and concatenate them with the latent vector
        label_embedding = self.label_embedding(labels)
        gen_input = torch.cat((noise, label_embedding), dim=1)

        # Transform the concatenated latent vector and labels to an 8x8x128 feature map
        out = self.fc1(gen_input)
        out = out.view(out.size(0), 128, 8, 8)

        # Pass the feature map through transpose convolutions to generate the final image
        img = self.conv_blocks(out)
        # print("Generator img shape: ", img.shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, classes=10, img_size=32, channels=3):
        super(Discriminator, self).__init__()

        self.img_size = img_size
        self.channels = channels

        # Label embedding to project class labels into image space
        self.label_embedding = nn.Embedding(classes, 50)

        # Label projection to image dimensions
        self.label_dense = nn.Sequential(
            nn.Linear(50, img_size * img_size),  # Convert label embedding to match image size (32x32)
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Convolutional blocks for downsampling the image
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(channels + 1, 128, kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),  # 8x8x128 -> 8192
            nn.Dropout(0.4),
            nn.Linear(128 * 8 * 8, 1),  # Final output is a single value (binary classification)
            nn.Sigmoid()  # Output is a probability (real or fake)
        )

    def forward(self, img, labels):
        
        # print("img shape: ", img.shape)
        # print("labels shape: ", labels.shape)

        # Embed the labels and project them into the image space
        label_embedding = self.label_embedding(labels)
        label_projection = self.label_dense(label_embedding)
        label_projection = label_projection.view(label_projection.size(0), 1, self.img_size, self.img_size)

        # print("label projection shape: ", label_projection.shape)

        # Concatenate the image with the label projection along the channel dimension
        d_in = torch.cat((img, label_projection), dim=1)

        # print("d_in shape: ", d_in.shape)

        # Pass the concatenated input through convolutional blocks
        validity = self.conv_blocks(d_in)
        return validity
