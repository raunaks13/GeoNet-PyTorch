#!/usr/bin/python3

import torch
import numpy as np
from imageio import imread
import random
import os
import cv2


def make_sequence_views(img_path, sequence_length, width):
    """Takes in concatenated target and source views, and returns each separately"""

    views = np.array(imread(img_path))
    w = views.shape[1]

    assert w == sequence_length*width
    
    tgt_view = np.array(views[:, width:width*2,: ])

    src_ids = [0, 2]
    src_views = [views[:, width*i:width*(i+1), :] for i in src_ids]
    
    # shape: (h, w, chnls)
    src_views = np.concatenate(src_views,axis=2)
    
    return torch.as_tensor(tgt_view), torch.as_tensor(src_views)

def make_intrinsics(cam_path):
    f = open(cam_path, 'r')
    intrinsics = np.array(f.readline().split()[0].split(',')).astype(np.float32).reshape(3, 3)
    return intrinsics

def make_resized_intrinsics_matrix(fx, fy, cx, cy):
    r1 = torch.unsqueeze(torch.tensor([fx, 0., cx]), dim=0)
    r2 = torch.unsqueeze(torch.tensor([0., fy, cy]), dim=0)
    r3 = torch.unsqueeze(torch.tensor([0., 0., 1.]), dim=0)
    intrinsics = torch.cat((r1, r2, r3), 0)
    return intrinsics

def data_augmentation(im, intrinsics, out_h, out_w):
    
    def random_scaling(im, intrinsics):
        """Inputs images with integer values"""
        in_h, in_w, _ = list(torch.as_tensor(im).size())
        scaling = torch.rand(2)*0.15 + 1    #Floating point random numbers from 1 to 1.15
        x_scaling = scaling[0]
        y_scaling = scaling[1]
        o_h = in_h * y_scaling
        o_w = in_w * x_scaling
        o_h = o_h.to(torch.int32, non_blocking=True)
        o_w = o_w.to(torch.int32, non_blocking=True)

        im = torch.as_tensor(cv2.resize(im.numpy(), (o_w, o_h), interpolation=cv2.INTER_AREA))

        fx = intrinsics[0, 0] * x_scaling
        fy = intrinsics[1, 1] * y_scaling
        cx = intrinsics[0, 2] * x_scaling
        cy = intrinsics[1, 2] * y_scaling
        intrinsics = make_resized_intrinsics_matrix(fx, fy, cx, cy)
        
        return im, intrinsics

    def random_cropping(im, intrinsics, out_h, out_w):
        in_h, in_w, _ = list(im.size())

        offset_y = np.random.randint(low=0, high=in_h-out_h+1)
        offset_x = np.random.randint(low=0, high=in_w-out_w+1)

        im = im[offset_y:offset_y+out_h, offset_x:offset_x+out_w, :]

        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2] - float(offset_x)
        cy = intrinsics[1, 2] - float(offset_y)
        
        intrinsics = make_resized_intrinsics_matrix(fx, fy, cx, cy)
        return im, intrinsics

    def random_coloring(im):
        in_h, in_w, in_c = list(im.size())
        im_f = im.to(torch.float32)
        im_f *= 1./255
        
        random_gamma = torch.rand(1)*0.4 + 0.8
        im_aug = im_f ** random_gamma

        random_brightness = torch.rand(1)*1.5 + 0.5
        im_aug = im_aug * random_brightness

        random_colors = torch.rand(in_c)*0.4 + 0.8
        white = torch.unsqueeze(torch.ones([in_h, in_w]), dim=2)
        color_image = torch.cat([white*random_colors[i] for i in range(in_c)], dim=2)
        im_aug *= color_image

        im_aug = torch.clamp(im_aug, min=0, max=1)
        im_aug *= 255
        im_aug = im_aug.to(torch.uint8, non_blocking=True)
        
        return im_aug

    im, intrinsics = random_scaling(im, intrinsics)
    im, intrinsics = random_cropping(im, intrinsics, out_h, out_w)

    im = torch.as_tensor(im).to(torch.uint8, non_blocking=True)
    #do_augment = torch.rand(1)
    #im = random_coloring(im) if do_augment > 0.5 else im

    return im, intrinsics

def get_multi_scale_intrinsics(intrinsics, num_scales):
    intrinsics_mscale = []
    for s in range(num_scales):
        fx = intrinsics[0,0]/(2 ** s)
        fy = intrinsics[1,1]/(2 ** s)
        cx = intrinsics[0,2]/(2 ** s)
        cy = intrinsics[1,2]/(2 ** s)
        intrinsics_mscale.append(make_resized_intrinsics_matrix(fx, fy, cx, cy))
    intrinsics_mscale = torch.cat(intrinsics_mscale, dim=1)
    return intrinsics_mscale

class testSequenceFolder(torch.utils.data.Dataset):
    
    def __init__(self, root, seed, split, sequence_length, img_width, img_height):
        np.random.seed(seed)
        random.seed(seed)
        self.root = root
        self.split = split

        if split == 'test':
            self.example_names = [self.root + name.split('.png\n')[0] for name in open('/ceph/raunaks/SIGNet/data/kitti/test_files_eigen.txt')]

        self.example_names = sorted(self.example_names)
        #random.shuffle(self.example_names)

        self.sequence_length = sequence_length
        self.width = img_width
        self.img_height = img_height

        self.imgs = [name + '.png' for name in self.example_names]

    def __getitem__(self, index):
        #return self.imgs[index]

        raw_im = np.array(imread(self.imgs[index]))
        # raw_im: Around (375, 1242, 3) for KITTI (single image data)
        
        scaled_im = torch.as_tensor(cv2.resize(raw_im, (self.width, self.img_height), interpolation=cv2.INTER_AREA))

        tgt_view = scaled_im.permute(2, 0, 1)
        return tgt_view
 
        #tgt_view = np.array(imread(self.imgs[index])).astype(np.float64)
        #tgt_view = np.moveaxis(tgt_view,-1,0)

        #return tgt_view

    def __len__(self):
        return len(self.example_names)

class SequenceFolder(torch.utils.data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/split/1.jpg
        root/split/1.cam
        root/split/2.jpg
        root/split/2.cam
        ...

        An image sequence is stacked along the horizontal dimension of a image,
        where the order is t-n,t-(n-1),...t-1,t,t+1,...,t+(n-1),t+n.
        Therefore, the length of the image sequence is 2*n+1.
        I_t is the tgt_view while the others are the src_views.

        The intrinsics correspnonding to an image sequence X is recorded inside the X.cam,
        with 9 numbers of the 3*3 intrinsics.

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, root, seed, split, sequence_length, img_width, img_height):
        
        np.random.seed(seed)
        random.seed(seed)
        self.root = root
        self.split = split
        """Takes filenames after '/' in root/split.txt and stores in example_names"""
        """
        self.root: /ceph/data/kitti_eigen_full/
        name: 2011_09_26_...._sync_02 01.jpg\n

        """
        if split == 'train':
            #self.example_names = [self.root + name.split('\n')[0].split(' ')[0] + '/' + name.split('\n')[0].split(' ')[1] for name in open('{}{}.txt'.format(self.root, 'train_lsd'))]
        
            self.example_names = [self.root + name.split('\n')[0].split(' ')[0] + '/' + name.split('\n')[0].split(' ')[1] for name in open('{}{}.txt'.format(self.root, self.split))]
        self.example_names = sorted(self.example_names)
        #random.shuffle(self.example_names)

        self.sequence_length = sequence_length
        self.width = img_width
        self.img_height = img_height
        #self.make_samples()

        self.imgs = [name + '.jpg' for name in self.example_names]
        self.cams = [name + '_cam.txt' for name in self.example_names]
        
        assert len(self.imgs) == len(self.cams)
        #print('Length: ' + str(len(self.imgs)) + '\n')
        #print(self.imgs, self.cams)

    def __getitem__(self, index):
        
        tgt_view, src_views = make_sequence_views(self.imgs[index], self.sequence_length, self.width)
        intrinsics = make_intrinsics(self.cams[index])
        """At this point images are of the shape (height, width, channels)"""
        
        image_all = torch.cat([tgt_view, src_views], dim=2)
        #image_all, intrinsics = data_augmentation(image_all, intrinsics, self.img_height, self.width)

        tgt_view = image_all[:, :, :3].permute(2, 0, 1)
        src_views = image_all[:, :, 3:].permute(2, 0, 1)
        #intrinsics = get_multi_scale_intrinsics(intrinsics, 4)

        sample = {'tgt_view': tgt_view, 'src_views': src_views, 'intrinsics': intrinsics}
        '''
        Sample Shapes:
        tgt_view: (channels, h, w)
        src_views: (channels, h, w)
        intrinsics: (3, 3)
        '''
        
        return tgt_view, src_views, intrinsics
        
    def __len__(self):
        return len(self.example_names)
