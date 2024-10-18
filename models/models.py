import torch
from torch import nn, optim


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
    def __init__(self, G, D, lr_G=2e-4, lr_D=2e-4, beta1=0.5, beta2=0.999, lambda_L1=100):
        super(ColorizationModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lambda_L1 = lambda_L1
        self.G = G.to(self.device)


        
        
        
