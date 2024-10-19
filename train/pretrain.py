import torch
import torch.optim as optim
import torch.nn as nn
from fastai.vision.learner import create_body
from torchvision.models import resnet18
from fastai.vision.models import unet
from config import *
from data.dataset import create_dataloader


def build_res_unet(n_input=1, n_output=2, size=IMG_SIZE):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resnet = resnet18(pretrained=True)
    body = create_body(resnet, pretrained=True, n_in=n_input, cut=-2)
    net = unet.DynamicUnet(body, n_output, (size, size)).to(device)
    return net

def pretrain(net, dl, opt, criterion, epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for epoch in range(epochs):
        for data in dl:  # Changed from 'train_dl' to 'dl'
            L, ab = data['L'].to(device), data['ab'].to(device)
            fake_ab = net(L)
            loss = criterion(fake_ab, ab)
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")


if __name__ == '__main__':
    print(TRAIN_PATH)
    train_dl = create_dataloader(data_path=TRAIN_PATH, split='train')
    net = build_res_unet()
    opt = optim.Adam(net.parameters(), lr=1e-4)
    criterion = nn.L1Loss()
    pretrain(net, train_dl, opt, criterion, PRETRAIN_EPOCHS)
    torch.save(net.state_dict(), 'pretrained_res18_unet.pth')
