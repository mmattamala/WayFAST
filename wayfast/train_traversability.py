#!/usr/bin/env python

import argparse
import os
import torch
import numpy as np
import cv2
from typing import Tuple
from tqdm import tqdm
from tqdm.auto import trange
from tqdm.contrib import tenumerate

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from models.resnet_unet import ResnetUnet
from models.resnet_depth_unet import ResnetDepthUnet
from utils.dataloader import TraversabilityDataset

import matplotlib.pyplot as plt


def wayfast_training(params):
    # Make folder for checkpoints
    os.makedirs(params.checkpoint_folder, exist_ok = True) 

    torch.manual_seed(params.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(params.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Make model
    if params.model == "rgb":
        net = ResnetUnet(params)
    elif params.model == "rgbd":
        net = ResnetDepthUnet(params)
    else:
        raise ValueError(f"Invalid model [{params.model}]. Valid options: rgb, rgbd")

    # Use to load a previously trained network
    if params.load_network_path is not None:
        print('Loading saved network from {}'.format(params.load_network_path))
        net.load_state_dict(torch.load(params.load_network_path))

    # Show number of GPU's being used
    print(f"Using {torch.cuda.device_count()} GPUs!")
    net = torch.nn.DataParallel(net).to(device)

    # Random input for testing
    random_rgb = torch.rand([2, 3, params.input_size[1], params.input_size[0]]).to(device)
    random_depth = torch.rand([2, 1, params.input_size[1], params.input_size[0]]).to(device)

    if params.model == "rgb":
        test = net(random_rgb)
    elif params.model == "rgbd":
        test = net(random_rgb, random_depth)
    else:
        raise ValueError(f"Invalid model [{params.model}]. Valid options: rgb, rgbd")
    print('test.shape:', test.shape)

    # Prepare dataset
    print("Loading dataset ...")
    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    dataset = TraversabilityDataset(params, transform)

    train_size, val_size = int(0.8*len(dataset)), np.ceil(0.2*len(dataset)).astype('int')
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader    = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2)
    test_loader     = DataLoader(val_dataset, batch_size=params.batch_size, shuffle=True, num_workers=2)

    print('Total loaded %d images' % len(dataset))
    print('Loaded %d train images' % train_size)
    print('Loaded %d valid images' % val_size)

    data = train_dataset[0]

    # Prepare optimizer and loss
    criterion = torch.nn.L1Loss(reduction='none')
    optimizer = torch.optim.Adam(net.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)

    print("Starting training")
    best_val_loss = np.inf
    train_loss_list = []
    val_loss_list = []
    for epoch in trange(params.epochs, desc='Training'):
        net.train()
        train_loss = 0.0
        
        for i, data in tenumerate(train_loader, desc='Inner'):
            data = (item.to(device).type(torch.float32) for item in data)
            color_img, depth_img, path_img, mu_img, nu_img, weight = data

            if params.model == "rgb":
                pred = net(color_img)
            elif params.model == "rgbd":
                pred = net(color_img, depth_img)
            else:
                raise ValueError(f"Invalid model [{params.model}]. Valid options: rgb, rgbd")

            label = mu_img

            loss = weight*criterion(pred*path_img, label)
            loss = torch.mean(loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)
            
        if (epoch) % 10 == 0:
            tqdm.write(f'Epoch [{epoch+1}/{params.epochs}], Loss: {train_loss}')
            tqdm.write(f'Learning Rate for this epoch: {optimizer.param_groups[0]["lr"]}')
        
        # evaluate the network on the test data
        with torch.no_grad():
            val_loss = 0.0
            net.eval()
            for i, data in enumerate(test_loader):
                data = (item.to(device).type(torch.float32) for item in data)
                color_img, depth_img, path_img, mu_img, nu_img, weight = data

                if params.model == "rgb":
                    pred = net(color_img)
                elif params.model == "rgbd":
                    pred = net(color_img, depth_img)
                else:
                    raise ValueError(f"Invalid model [{params.model}]. Valid options: rgb, rgbd")

                label = mu_img

                loss = weight*criterion(pred*path_img, label)
                loss = torch.mean(loss)

                val_loss += loss.item()
            val_loss /= len(test_loader)
            val_loss_list.append(val_loss)

        if (epoch + 1) % 5 == 0:
            inverse_transform = transforms.Compose([
                transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1/0.229, 1/0.224, 1/0.225]),
                transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1.0, 1.0, 1.0]),
                ])
            color_out = inverse_transform(color_img)[0].permute(1, 2, 0).cpu().numpy()
            color_out = np.interp(color_out, (color_out.min(), color_out.max()), (0, 255))

            pred_out = np.clip(255*pred[0,0,:,:].detach().cpu().numpy(), 0, 255)

            # Write input color
            cv2.imwrite(f"{params.checkpoint_folder}/epoch{epoch:03}_color.jpg", color_out)
            cv2.imwrite(f"{params.checkpoint_folder}/epoch{epoch:03}_pred.jpg", pred_out)
        
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            tqdm.write(f'Updating best validation loss: {best_val_loss}')
            torch.save(net.module.state_dict(),f'{params.checkpoint_folder}/best_predictor_depth.pth')

        # Save final model
        torch.save(net.module.state_dict(),f'{params.checkpoint_folder}/predictor_depth.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""WayFAST training script""")
    parser.add_argument("--data_path", required=False, type=str, default="data_test")
    parser.add_argument("--checkpoint_folder", required=False, type=str, default="checkpoints")
    parser.add_argument("--model", required=False, type=str, default="rgb")
    parser.add_argument("--preproc", required=False, type=bool, default=True)
    parser.add_argument("--depth_mean", required=False, type=float, default=3.5235)
    parser.add_argument("--depth_std", required=False, type=float, default=10.6645)
    parser.add_argument("--seed", required=False, type=int, default=230)
    parser.add_argument("--epochs", required=False, type=int, default=50)
    parser.add_argument("--batch_size", required=False, type=int, default=16)
    parser.add_argument("--learning_rate", required=False, type=float, default=1e-4)
    parser.add_argument("--weight_decay", required=False, type=float, default=1e-5)
    parser.add_argument("--pretrained", required=False, type=bool, default=True)
    parser.add_argument("--load_network_path", required=False, type=int, default=None)
    parser.add_argument("--input_size", required=False, type=Tuple[int,int], default=(424, 240))
    parser.add_argument("--output_size", required=False, type=Tuple[int,int], default=(424, 240))
    parser.add_argument("--output_channels", required=False, type=int, default=1)
    parser.add_argument("--bottleneck_dim", required=False, type=int, default=256)
    
    args = parser.parse_args()

    wayfast_training(args)

