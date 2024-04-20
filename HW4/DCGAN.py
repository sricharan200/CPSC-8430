#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torchvision.utils as utils
import matplotlib.animation as animation
from IPython.display import HTML
import time
from torch.utils.data import Subset
import torchvision.models as models
import torch.nn.functional as F
from scipy import linalg
import pandas as pd


# In[2]:


torch.manual_seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:',device)
learning_rate = 2e-4
batch_siz = 128
image_size= 64
channel_image = 3  
noise_dim = 100
Total_epochs = 40
feat_disc = 64 
feat_gen = 64 
beta = 0.5


# In[3]:


dataset = datasets.CIFAR10(root="CIFAR10data", download=True,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_siz,
                                         shuffle=True, num_workers=2)


# In[4]:


real_batch = next(iter(dataloader))
plt.figure(figsize=(10,10))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(utils.make_grid(real_batch[0].to(device)[:32], padding=2, normalize=True).cpu(),(1,2,0)))


# In[5]:


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            self.Dnet(features_d, features_d * 2, 4, 2, 1),      
            self.Dnet(features_d * 2, features_d * 4, 4, 2, 1),  
            self.Dnet(features_d * 4, features_d * 8, 4, 2, 1),  
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),
        )
    def Dnet(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


# In[6]:


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            self.Gnet(channels_noise, features_g * 16, 4, 1, 0), 
            self.Gnet(features_g * 16, features_g * 8, 4, 2, 1), 
            self.Gnet(features_g * 8, features_g * 4, 4, 2, 1),  
            self.Gnet(features_g * 4, features_g * 2, 4, 2, 1),  
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )
    def Gnet(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels,momentum=0.9),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)


# In[7]:


def weights_initialization(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


# In[8]:


gen = Generator(noise_dim, channel_image, feat_gen).to(device)
disc = Discriminator(channel_image, feat_disc).to(device)
weights_initialization(gen)
weights_initialization(disc)


# In[9]:


optimizer_Gen = optim.Adam(gen.parameters(), lr=learning_rate, betas=(beta, 0.999))
optimizer_Disc = optim.Adam(disc.parameters(), lr=learning_rate, betas=(beta, 0.999))
criterion = nn.BCELoss()
fixed_noise = torch.randn(32, noise_dim, 1, 1).to(device)  
step = 0


# In[10]:


gen.train(),disc.train()


# In[11]:


class InceptionV3(nn.Module):

    DEFAULT_BLOCK_INDEX = 3

    BLOCK_INDEX_BY_DIM = {
        64: 0,   # First max pooling features
        192: 1,  # Second max pooling featurs
        768: 2,  # Pre-aux classifier features
        2048: 3  # Final average pooling features
    }

    def __init__(self,
                 output_blocks=[DEFAULT_BLOCK_INDEX],
                 resize_input=True,
                 normalize_input=True,
                 requires_grad=False):
        
        super(InceptionV3, self).__init__()

        self.resize_input = resize_input
        self.normalize_input = normalize_input
        self.output_blocks = sorted(output_blocks)
        self.last_needed_block = max(output_blocks)

        assert self.last_needed_block <= 3,             'Last possible output block index is 3'

        self.blocks = nn.ModuleList()

        
        inception = models.inception_v3(pretrained=True)

        # Block 0: input to maxpool1
        block0 = [
            inception.Conv2d_1a_3x3,
            inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2)
        ]
        self.blocks.append(nn.Sequential(*block0))

        # Block 1: maxpool1 to maxpool2
        if self.last_needed_block >= 1:
            block1 = [
                inception.Conv2d_3b_1x1,
                inception.Conv2d_4a_3x3,
                nn.MaxPool2d(kernel_size=3, stride=2)
            ]
            self.blocks.append(nn.Sequential(*block1))

        # Block 2: maxpool2 to aux classifier
        if self.last_needed_block >= 2:
            block2 = [
                inception.Mixed_5b,
                inception.Mixed_5c,
                inception.Mixed_5d,
                inception.Mixed_6a,
                inception.Mixed_6b,
                inception.Mixed_6c,
                inception.Mixed_6d,
                inception.Mixed_6e,
            ]
            self.blocks.append(nn.Sequential(*block2))

        # Block 3: aux classifier to final avgpool
        if self.last_needed_block >= 3:
            block3 = [
                inception.Mixed_7a,
                inception.Mixed_7b,
                inception.Mixed_7c,
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
            ]
            self.blocks.append(nn.Sequential(*block3))

        for param in self.parameters():
            param.requires_grad = requires_grad

    def forward(self, inp):
        outp = []
        x = inp

        if self.resize_input:
            x = F.interpolate(x,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)

        if self.normalize_input:
            x = 2 * x - 1  # Scale from range (0, 1) to range (-1, 1)

        for idx, block in enumerate(self.blocks):
            x = block(x)
            if idx in self.output_blocks:
                outp.append(x)

            if idx == self.last_needed_block:
                break

        return outp
    
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
model = InceptionV3([block_idx])
model=model.cuda()


# In[12]:


def calc_activation_statistics(images,model,batch_siz=128, dims=2048,
                    cuda=False):
    model.eval()
    act=np.empty((len(images), dims))
    
    if cuda:
        batch=images.cuda()
    else:
        batch=images
    pred = model(batch)[0]
    if pred.size(2) != 1 or pred.size(3) != 1:
        pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

    act= pred.cpu().data.numpy().reshape(pred.size(0), -1)
    
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


# In[13]:


def calc_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape,         'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape,         'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


# In[14]:


def calc_fretchet(images_real,images_fake,model):
    mu_1,std_1=calc_activation_statistics(images_real,model,cuda=True)
    mu_2,std_2=calc_activation_statistics(images_fake,model,cuda=True)
    fid_value = calc_frechet_distance(mu_1, std_1, mu_2, std_2)
    return fid_value


# In[15]:


GenLoss = []
DiscLoss = []
image_array = []
FID_array = []
iterations = 0
time_start = time.time() 
for epoch in range(Total_epochs):
    epoch = epoch+1
    for batch_id, data in enumerate(dataloader,0):
        true = data[0].to(device)
        noise = torch.randn(batch_siz, noise_dim, 1, 1).to(device)
        false = gen(noise) 

        disc_true = disc(true).reshape(-1)
        loss_disc_true = criterion(disc_true, torch.ones_like(disc_true))
        disc_false = disc(false.detach()).reshape(-1)
        loss_disc_false = criterion(disc_false, torch.zeros_like(disc_false))
        loss_disc = (loss_disc_true + loss_disc_false) / 2
        disc.zero_grad()
        loss_disc.backward()
        optimizer_Disc.step()

        output = disc(false).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        optimizer_Gen.step()
        GenLoss.append(loss_gen.detach().cpu())
        DiscLoss.append(loss_disc.detach().cpu())
        if (iterations % 500 == 0) or ((epoch == Total_epochs) and (batch_id == len(dataloader)-1)):
            with torch.no_grad():
                false = gen(fixed_noise).detach().cpu()
            image_array.append(utils.make_grid(false, padding=2, normalize=True))
        iterations += 1
    fretchet_dist=calc_fretchet(true,false,model)
    FID_array.append(fretchet_dist)
    if epoch%5 == 0:
        print( f'\nEpoch: [{epoch}/{Total_epochs}]Batch: {batch_id}/{len(dataloader)}                   Loss Discriminator: {loss_disc:.3f}, Loss Generator: {loss_gen:.3f} FID:{fretchet_dist:.3f} ') #,end = '\r', flush=True)    
    images = gen(fixed_noise)     
time_end = time.time()
print('\n\n elapsed time：%.2f s.'%(time_end-time_start))


# In[22]:


plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss While Training")
plt.plot(GenLoss,label="Gen")
plt.plot(DiscLoss,label="Disc")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend() 
plt.show()


# In[17]:


real_batch = next(iter(dataloader))
plt.figure(figsize=(15,15))
plt.subplot(1,2,1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(utils.make_grid(real_batch[0].to(device)[:32], padding=5, normalize=True).cpu(),(1,2,0)))
plt.subplot(1,2,2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(image_array[-1],(1,2,0)))
plt.show()


# In[21]:


plt.figure(figsize=(10,5))
plt.title("FID Scores of DCGAN")
plt.plot(FID_array,label="DCGAN")
plt.xlabel("Epochs")
plt.ylabel("FID")
plt.legend()
plt.show()


# In[ ]:





# In[19]:


np.mean(GenLoss),np.mean(DiscLoss),np.min(GenLoss),np.min(DiscLoss),GenLoss[-1],DiscLoss[-1]


# In[20]:


np.min(FID_array), np.max(FID_array), np.mean(FID_array), FID_array[-1]


# In[ ]:





# In[ ]:




