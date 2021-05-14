import torch
import config
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self,img_channels = config.IMG_CHANNELS,featureDim = config.FEATURE_DIM, zDim = config.ZDIM):
        super(VAE,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=img_channels,out_channels=16,kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=5)
        self.enFC1 = nn.Linear(featureDim,zDim)
        self.enFC2 = nn.Linear(featureDim,zDim)

        self.deFC = nn.Linear(zDim,featureDim)
        self.deconv1 = nn.ConvTranspose2d(32,16,5)
        self.deconv2 = nn.ConvTranspose2d(16,img_channels,5)
    
    def encoder(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,config.FEATURE_DIM)
        # The output feature map are fed into 2 fully-connected layers to predict mean (pm) and variance (logVar)
        # pm and logVar are used for generating middle representation z and KL divergence loss
        pm = self.enFC1(x)
        logVar = self.enFC2(x)
        return pm, logVar
    
    def reparameterize(self,pm,logVar):
        #Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2)
        eps = torch.randn_like(std)
        return pm + std*eps
    
    def decoder(self,z):
        # z is fed back into a fully-connected layers and then into two transpose convolutional layers
        # The generated output is the same size of the original input
        x = F.relu(self.deFC(z))
        x = x.view(-1,32,20,20)
        x = F.relu(self.deconv1(x))
        x = torch.sigmoid(self.deconv2(x))
        return x
    
    def forward(self,x):
        pm, logVar = self.encoder(x)
        z = self.reparameterize(pm,logVar)
        out = self.decoder(z)
        return out,pm,logVar