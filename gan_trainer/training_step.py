"""
One-step training functions for different models
"""
import numpy as np

import torch
from torch.autograd import Variable
from utils.gan_loss import generator_loss, discriminator_loss

def classifier_train_step(classifieur, inputs, optimizer, criterion, labels):
    """Summary

    Args:
        classifieur (TYPE)
        inputs (TYPE)
        optimizer (TYPE)
        criterion (TYPE)
        labels (TYPE)
        classifieur (TYPE)
        inputs (TYPE)
        optimizer (TYPE)
        criterion (TYPE)
        labels (TYPE)

    Returns:
        TYPE: loss value
    """
    optimizer.zero_grad()
    outputs = classifieur(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


def generator_train_step(discriminator, generator, g_optimizer, criterion,
                         batch_size,  latent_dim, labels=None, n_classes=None):
    """Summary

    Args:
        discriminator (TYPE)
        generator (TYPE)
        g_optimizer (TYPE)
        criterion (TYPE)
        batch_size (int)
        latent_dim (int)
        labels (None, optional)
        n_classes (int, optional)

    Returns:
        TYPE: loss value
    """
    g_optimizer.zero_grad()

    device = next(generator.parameters()).device

    generator_input = Variable(torch.randn(batch_size, latent_dim)).to(device)
    # If no labels are given we generate random labels
    if labels is None:
        assert isinstance(n_classes, int), 'n_classes must be of type int when labels are not given'
        labels = Variable(torch.LongTensor(np.random.randint(0, n_classes, batch_size))).to(device)
    # print("labels shape from generator_train_step")
    # print(labels.shape)
    fake_images = generator(generator_input, labels)

    validity = discriminator(fake_images, labels).squeeze(dim=1)

    # print("fake_images shape", fake_images.shape)
    # print("validity shape ", validity.shape)
    # g_loss = criterion(validity, Variable(torch.ones(batch_size)).to(device))

    g_loss = generator_loss(validity)

    # print("g_loss shape" , g_loss.shape)
    # print("g_loss = ", g_loss)
    g_loss.backward()
    g_optimizer.step()
    return g_loss.item()


def discriminator_train_step(discriminator, generator, d_optimizer, criterion,
                             real_images, labels, latent_dim, n_classes):
    """
    Args:
        discriminator (TYPE): 
        generator (TYPE): 
        d_optimizer (TYPE): 
        criterion (TYPE): 
        real_images (TYPE): 
        labels (TYPE): 
        latent_dim (int): 
        n_classes (int): 
    
    Returns:
        TYPE: loss value
    """
    d_optimizer.zero_grad()

    device = next(generator.parameters()).device
    batch_size = len(real_images)

    # train with real images
    real_validity = discriminator(real_images, labels).squeeze(dim=1)

    # print("chnage check")
    # loss for real images
    # d_loss = criterion(real_validity, Variable(torch.ones(batch_size)).to(device))

    # train with fake images
    generator_input = Variable(torch.randn(batch_size, latent_dim)).to(device)
    fake_labels = Variable(torch.LongTensor(np.random.randint(0, n_classes, batch_size))).to(device)
    fake_images = generator(generator_input, fake_labels)
    fake_validity = discriminator(fake_images, fake_labels).squeeze(dim=1)

    # print("real image shape: ", real_images.shape)
    # print("real validity shape: ", real_validity.shape)
    # print("fake image shape: ", fake_images.shape)
    # print("fake validity shape: ", fake_validity.shape)

    # loss for fake images
    # d_loss += criterion(fake_validity, Variable(torch.zeros(batch_size)).to(device))

    d_loss_real, d_loss_fake = discriminator_loss(fake_validity, real_validity)
    d_loss = d_loss_fake + d_loss_real

    # print("d_loss shape: " , d_loss.shape)
    # print("d_loss = ", d_loss)

    d_loss.backward()
    d_optimizer.step()
    return d_loss.item()
