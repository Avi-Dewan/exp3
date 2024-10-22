"""
Pretraining functions for classifier and GAN
"""
import os

import numpy as np
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.nn.parameter import Parameter

from torchvision.utils import save_image

from gan_trainer.training_step import classifier_train_step, generator_train_step, discriminator_train_step
from modules.module import feat2prob, target_distribution 

from models.resnet import ResNet, BasicBlock
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from utils.util import cluster_acc, Identity, AverageMeter, seed_torch, str2bool, toggle_grad , prepare_z_y

def test(model, test_loader, args, tsne=False):
    model.eval()
    preds=np.array([])
    targets=np.array([])
    feats = np.zeros((len(test_loader.dataset), args.n_clusters))
    probs= np.zeros((len(test_loader.dataset), args.n_clusters))
    device = next(model.parameters()).device
    for batch_idx, (x, label, idx) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        feat = model(x)
        prob = feat2prob(feat, model.center)
        _, pred = prob.max(1)
        targets=np.append(targets, label.cpu().numpy())
        preds=np.append(preds, pred.cpu().numpy())
        idx = idx.data.cpu().numpy()
        feats[idx, :] = feat.cpu().detach().numpy()
        probs[idx, :] = prob.cpu().detach().numpy()
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds)
    print('Pretrained Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    probs = torch.from_numpy(probs)

    if tsne:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt
        # tsne plot
         # Create t-SNE visualization
        X_embedded = TSNE(n_components=2).fit_transform(feats)  # Use meaningful features for t-SNE

        plt.figure(figsize=(8, 6))
        plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=targets, cmap='viridis')
        plt.title("t-SNE Visualization of Learned Features on Unlabelled CIFAR-10 Subset")
        plt.savefig(args.model_folder+'/tsne.png')
    return acc, nmi, ari, probs

def getPsedoLabels(model, train_loader, args):
    model.eval()
    pseudoLabels=np.array([])
    targets=np.array([])

    device = next(model.parameters()).device
    for batch_idx, ((x, _), label, _) in enumerate(tqdm(train_loader)):
        x = x.to(device)
        feat = model(x)
        prob = feat2prob(feat, model.center)
        _, pseudoLabel = prob.max(1)
        pseudoLabels=np.append(pseudoLabels, pseudoLabel.cpu().numpy())
        targets=np.append(targets, label.cpu().numpy())

    acc, nmi, ari = cluster_acc(targets.astype(int), pseudoLabels.astype(int)), nmi_score(targets, pseudoLabels), ari_score(targets, pseudoLabels)
    print('PseudoLabel acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))

    return pseudoLabels

def classifier_pretraining(args, train_loader, eval_loader):
    # Classifier pretraining on source data
    model_dict = torch.load(args.cls_pretraining_path, map_location=args.device)
    model = ResNet(BasicBlock, [2,2,2,2], args.n_classes).to(args.device)
    model.load_state_dict(model_dict['state_dict'], strict=False)
    model.center = Parameter(model_dict['center'])
        
    return model


def gan_pretraining(generator, discriminator, classifier, loader_train,
                    lr_g, lr_d, latent_dim, n_classes, n_epochs,
                    img_size, results_path):
    """
    Args:
        generator (TYPE)
        discriminator (TYPE)
        classifier (TYPE)
        loader_train (TYPE)
        lr_g (TYPE)
        lr_d (TYPE)
        latent_dim (TYPE)
        n_classes (TYPE)
        n_epochs (TYPE)
        img_size (TYPE)
        results_path (TYPE)

    Returns:
        Pretrained generator and discriminator
    """
    img_pretraining_path = ''.join([results_path, '/images'])
    models_pretraining_path = ''.join([results_path, '/gan_models'])

    g_pretrained = ''.join([results_path, '/generator_pretrained.pth'])
    d_pretrained = ''.join([results_path, '/discriminator_pretrained.pth'])

    device = next(classifier.parameters()).device

    criterion_gan = nn.BCELoss().to(device)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr_d)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=lr_g)

    os.makedirs(img_pretraining_path, exist_ok=True)
    os.makedirs(models_pretraining_path, exist_ok=True)

    loaded_gen = False
    loaded_dis = False

    if os.path.isfile(g_pretrained):
        generator.load_state_dict(torch.load(g_pretrained))
        print('loaded existing generator')
        loaded_gen = True

    if os.path.isfile(d_pretrained):
        discriminator.load_state_dict(torch.load(d_pretrained))
        print('loaded existing discriminator')
        loaded_dis = True
    
    z_, y_ = prepare_z_y(G_batch_size=128, dim_z=latent_dim, nclasses=n_classes)


    if not(loaded_gen and loaded_dis):
        print('Starting Pre Training GAN')
        g_loss_epochs = []
        d_loss_epochs = []
        for epoch in range(n_epochs):
            print(f'Starting epoch {epoch}/{n_epochs}...', end=' ')
            generator.train()
            discriminator.train()

            g_loss_list = []
            d_loss_list = []
            
            for batch_idx, ((images, _), targets, _) in enumerate(tqdm(loader_train)):

                # real_images = Variable(images).to(device)
                # _, labels = torch.max(classifier(real_images), dim=1)

                # real_images = Variable(images).to(device)
                # feat = classifier(real_images)
                # prob = feat2prob(feat, classifier.center)
                # _, labels = prob.max(1)

                real_images = Variable(images).to(device)
                labels = (targets - 5).to(device)

                toggle_grad(generator, False)
                toggle_grad(discriminator, True)

                d_loss = discriminator_train_step(discriminator, generator, d_optimizer, 
                                                  real_images, labels, z_, y_)
                d_loss_list.append(d_loss)

                toggle_grad(generator, True)
                toggle_grad(discriminator, False)

                g_loss = generator_train_step(discriminator, generator, g_optimizer,
                                              z_, y_)
                g_loss_list.append(g_loss)

            generator.eval()

            # Number of images per class
            n_images_per_class = 5
            z_.sample_()
            # Generate latent space and labels for each class
            latent_space = Variable(torch.randn(n_classes * n_images_per_class, latent_dim)).to(device)
            gen_labels = Variable(torch.LongTensor(np.repeat(np.arange(n_classes), n_images_per_class))).to(device)

            # Convert labels to one-hot encoding
            gen_labels_one_hot = torch.nn.functional.one_hot(gen_labels, num_classes=n_classes).float().to(device)

            # Generate images
            gen_imgs = generator(latent_space, gen_labels_one_hot).view(-1, 3, img_size, img_size)

            # Print losses
            print(f"[D loss: {np.mean(d_loss_list)}] [G loss: {np.mean(g_loss_list)}]")
            g_loss_epochs.append(np.mean(g_loss_list))
            d_loss_epochs.append(np.mean(d_loss_list))

            if epoch == n_epochs - 1:
                # Ensure images are in the correct format for saving
                gen_imgs = gen_imgs.float().cpu()
                
                # Save generated images
                save_image(gen_imgs, img_pretraining_path + f'/epoch_{epoch:02d}.png', nrow=n_images_per_class, normalize=True)
                
                # Save model states
                torch.save(generator.state_dict(), models_pretraining_path + f'/{epoch:02d}_gen.pth')
                torch.save(discriminator.state_dict(), models_pretraining_path + f'/{epoch:02d}_dis.pth')
                
                # Plot and save loss graph
                plt.figure(figsize=(10, 5))
                plt.title("Generator and Discriminator Loss During Training")
                plt.plot(g_loss_epochs, label="G_loss")
                plt.plot(d_loss_epochs, label="D_loss")
                plt.xlabel("epochs")
                plt.ylabel("Loss")
                plt.legend()
                plt.savefig(os.path.join(img_pretraining_path, 'gan_loss.png'))

        print('Finished Pre Training GAN')
        print('\n')


    return generator, discriminator
