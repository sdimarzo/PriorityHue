import torch
from torch import nn, optim
from utils.init_weights import init_weights
from models.GANLoss import GANLoss
from config import *

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels, num_filters=64, n_down=3):
        super().__init__()
        model = [self.get_layers(in_channels, num_filters, norm=False)]
        model += [self.get_layers(num_filters * 2 ** i, num_filters * 2 ** (i + 1), s=1 if i == (n_down - 1) else 2)
                  for i in range(n_down)]  
        model += [self.get_layers(num_filters * 2 ** n_down, 1, s=1, norm=False,
                                  act=False)]  
        self.model = nn.Sequential(*model)

    def get_layers(self, ni, nf, k=4, s=2, p=1, norm=True,
                   act=True):  
        layers = [
            nn.Conv2d(ni, nf, k, s, p, bias=not norm)] 
        if norm: layers += [nn.BatchNorm2d(nf)]
        if act: layers += [nn.LeakyReLU(0.2, True)]
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class ColorizationModel(nn.Module):
    def __init__(self, G, lr_G=LR_G, lr_D=LR_D, beta1=BETA1, beta2=BETA2, lambda_L1=LAMBDA_L1):
        super(ColorizationModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lambda_L1 = lambda_L1
        self.G = G.to(self.device)
        self.D = PatchDiscriminator(in_channels=3, num_filters=64).to(self.device)
        self.D = init_weights(self.D)
        self.GANCriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=lr_D, betas=(beta1, beta2))   
        
    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

    def set_requires_grad(self, model, requires_grad=True):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def forward(self):
        self.fake_color = self.G(self.L)
        
    def backward_D(self):
        fake_img = torch.cat([self.L, self.fake_color], 1)
        pred_fake = self.D(fake_img.detach())
        self.loss_D_fake = self.GANCriterion(pred_fake, False)
        real_img = torch.cat([self.L, self.ab], 1)
        pred_real = self.D(real_img)
        self.loss_D_real = self.GANCriterion(pred_real, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()
        
    def backward_G(self):
        fake_img = torch.cat([self.L, self.fake_color], 1)
        pred_fake = self.D(fake_img)
        self.loss_G_GAN = self.GANCriterion(pred_fake, True)
        self.loss_G_L1 = self.lambda_L1 * nn.functional.l1_loss(self.fake_color, self.ab)
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()
        
    def optimize_parameters(self):
        self.forward()
        self.D.train()
        self.set_requires_grad(self.D, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        
        self.G.train()
        self.set_requires_grad(self.G, True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        
        
        
        
        
        
        
        
        
        
        
