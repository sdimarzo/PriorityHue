import torch
from torch import nn, optim
from utils.init_weights import init_weights
from GANLoss import GANLoss
from config import *

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_layers=8, num_filters=64):
        super(PatchDiscriminator, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = 4
        self.padding = 1
        self.norm_layer = nn.BatchNorm2d()
        sequence = [nn.Conv2d(self.in_channels, self.num_filters, kernel_size=self.kernel_size, stride=2, padding=self.padding)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, num_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(num_filters * nf_mult_prev, num_filters * nf_mult, kernel_size=self.kernel_size, stride=2, padding=self.padding, bias=nn.InstanceNorm2d),
                self.norm_layer(num_filters * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** num_layers, 8)
        sequence += [
            nn.Conv2d(self.num_filters * nf_mult_prev, self.num_filters * nf_mult, kernel_size=self.kernel_size, stride=1, padding=self.padding, bias=nn.InstanceNorm2d),
            self.norm_layer(self.num_filters * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(self.num_filters * nf_mult, 1, kernel_size=self.kernel_size, stride=1, padding=self.padding)]  # output 1 channel prediction map

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class ColorizationModel(nn.Module):
    def __init__(self, G, D, lr_G=LR_G, lr_D=LR_D, beta1=BETA1, beta2=BETA2, lambda_L1=LAMBDA_L1):
        super(ColorizationModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lambda_L1 = lambda_L1
        self.G = G.to(self.device)
        self.D = PatchDiscriminator(in_channels=1, out_channels=1, num_layers=8, num_filters=64).to(self.device)
        self.D = init_weights(self.D)
        self.GANCriterion = GANLoss(gan_mode='vanilla').to(self.device)
        self.optimizer_G = optim.Adam(self.G.parameters(), lr=lr_G, betas=(beta1, beta2))
        self.optimizer_D = optim.Adam(self.D.parameters(), lr=lr_D, betas=(beta1, beta2))   
        
    def setup_input(self, data):
        self.L = data['L'].to(self.device)
        self.ab = data['ab'].to(self.device)

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
        self.set_requires_grad([self.D], True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        
        self.G.train()
        self.set_requires_grad([self.G], True)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        
        
        
        
        
        
        
        
        
        
        
        
