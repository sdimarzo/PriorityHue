import torch
import os
from models.models import ColorizationModel
from data.dataset import create_dataloader
from train.pretrain import build_res_unet
from config import *

def train(model, dataloader, epochs, start_epoch=0, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for e in range(start_epoch, epochs):
        epoch_loss = 0.0
        for idx, data in enumerate(dataloader):
            model.setup_input(data)
            model.optimize_parameters()
            
            # Calculate and accumulate loss
            current_loss = model.loss_G.item() + model.loss_D.item()
            epoch_loss += current_loss
            
            if idx % 10 == 0:  # Print every 10 batches
                print(f'Epoch [{e+1}/{epochs}], Step [{idx+1}/{len(dataloader)}], Loss: {current_loss:.4f}')
        
        # Print epoch average loss
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch [{e+1}/{epochs}] completed. Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint every 5 epochs
        if (e + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{e+1}.pth')
            torch.save({
                'epoch': e + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_G_state_dict': model.optimizer_G.state_dict(),
                'optimizer_D_state_dict': model.optimizer_D.state_dict(),
                'loss_G': model.loss_G,
                'loss_D': model.loss_D,
            }, checkpoint_path)
            print(f'Checkpoint saved: {checkpoint_path}')
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_G_state_dict': model.optimizer_G.state_dict(),
        'optimizer_D_state_dict': model.optimizer_D.state_dict(),
    }, final_model_path)
    print(f'Final model saved: {final_model_path}')

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dl = create_dataloader(data_path=TRAIN_PATH, split='train')
    netG = build_res_unet(n_input=1, n_output=2, size=IMG_SIZE)
    netG.load_state_dict(torch.load("./checkpoints/pretrained_res18_unet.pth", map_location=device))
    model = ColorizationModel(G=netG)

    # Find the latest checkpoint
    checkpoints = [f for f in os.listdir('checkpoints') if f.startswith('checkpoint_epoch_')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        checkpoint_path = os.path.join('checkpoints', latest_checkpoint)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            model.optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            start_epoch = checkpoint['epoch']
            print(f"Resuming training from epoch {start_epoch}")
        except RuntimeError as e:
            print(f"Error loading checkpoint: {e}")
            print(f"Checkpoint file might be corrupted: {checkpoint_path}")
            print("Starting training from scratch")
            start_epoch = 0
    else:
        start_epoch = 0
        print("Starting training from scratch")

    # Update EPOCHS in config.py to the desired total number of epochs
    train(model, train_dl, EPOCHS, start_epoch)
