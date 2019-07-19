import torch
from utils_edited import *

def image_similarity(alpha,x,y):
    # print('alpha*DSSIM(x,y): {:.16f}\n torch.abs(x-y): {:.16f}'.format(torch.mean(alpha*DSSIM(x,y)),torch.mean((1-alpha)*torch.abs(x-y))))
    return alpha * DSSIM(x,y) + (1-alpha) * torch.abs(x - y)

def smooth_loss(depth,image):
    # depth: (12, h, w, 1)
    # image: (12, h, w, 3)
    
    gradient_depth_x = gradient_x(depth)  # (TODO)shape: bs,h,w,1
    gradient_depth_y = gradient_y(depth)

    gradient_img_x = gradient_x(image)  # (TODO)shape: bs,h,w,3
    gradient_img_y = gradient_y(image)

    exp_gradient_img_x = torch.exp(-torch.mean(torch.abs(gradient_img_x), 3, True)) # (TODO)shape: bs,h,w,1
    exp_gradient_img_y = torch.exp(-torch.mean(torch.abs(gradient_img_y), 3, True)) 

    smooth_x = gradient_depth_x*exp_gradient_img_x
    smooth_y = gradient_depth_y*exp_gradient_img_y

    return torch.mean(torch.abs(smooth_x))+torch.mean(torch.abs(smooth_y))

def flow_smooth_loss(flow,img):
    # TODO two flows ?= rigid flow + object motion flow
    smoothness = 0
    for i in range(2):
        # TODO shape of flow: bs,channels(2),h,w
        smoothness += smooth_loss(flow[:, i, :, :].unsqueeze(1), img)
    return smoothness/2
