#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

### ADDED: Helper function to load a mask image and convert to a tensor
import cv2
import numpy as np

def load_mask_tensor(path, height, width):
    """ Loads a mask image, resizes it, and converts to a PyTorch tensor. """
    try:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        
        # Resize mask to match image dimensions if necessary
        if mask.shape[0] != height or mask.shape[1] != width:
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
            
        # Normalize to [0, 1] and convert to tensor
        mask = torch.from_numpy(mask.astype(np.float32) / 255.0)
        
        # Add a channel dimension to make it [1, H, W]
        return mask.unsqueeze(0)
    except Exception as e:
        print(f"Error loading mask {path}: {e}")
        return None