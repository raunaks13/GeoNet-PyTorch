from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.backends.cudnn as cudnn
import torch.optim
import DispNetS
import FlowNet
import PoseNet
#import DispUnet
from sequence_folders import SequenceFolder
from sequence_folders import testSequenceFolder
from loss_functions import *
from utils_edited import *
import time
import os
from tensorboardX import SummaryWriter

global n_iter
n_iter = 0

class GeoNetModel(object):
    def __init__(self, args, device):
        self.args = args
        
        # Nets preparation
        self.disp_net = DispNetS.DispNetS()
        self.pose_net = PoseNet.PoseNet(args.num_source)
        
        
        # input channels: src_views * (3 tgt_rgb + 3 src_rgb + 3 warp_rgb + 2 flow_xy +1 error )
        #self.flow_net = FlowNet.FlowNet(12, self.config['flow_scale_factor'])

        if device.type == 'cuda':
            self.disp_net.cuda()
            self.pose_net.cuda()
            #self.flow_net.cuda()

        #Weight initialization
        if (not args.train_flow) and args.is_train:
            print('Initializing weights from scratch')
            self.disp_net.init_weight()
            self.pose_net.init_weight()

        if not args.is_train:
            
            if not os.path.exists(args.ckpt_dir):
                os.makedirs(args.ckpt_dir)
                
            path = '{}/{}_{}'.format(args.ckpt_dir, 'rigid_depth', str(args.ckpt_index) + '.pth')
            print('Loading saved model weights from {}'.format(path))
            ckpt = torch.load(path)
            self.disp_net.load_state_dict(ckpt['disp_net_state_dict'])
            self.pose_net.load_state_dict(ckpt['pose_net_state_dict'])

        """
        else:
            ckpt = torch.load(config['ckpt_path'])
            self.disp_net.load_state_dict(ckpt['disp_net_state_dict'])
            self.pose_net.load_state_dict(ckpt['pose_net_state_dict'])
            if train_flow:
                if 'flow_net_state_dict' in ckpt:
                    self.flow_net.load_state_dict(ckpt['flow_net_state_dict'])
                else:
                    self.flow_net.init_weight()
        """

        #cudnn.benchmark = True
        # for multiple GPUs
        #self.disp_net = torch.nn.DataParallel(self.disp_net)
        #self.pose_net = torch.nn.DataParallel(self.pose_net)

        self.nets = {
            'disp': self.disp_net,
            'pose': self.pose_net
            #'flow': self.flow_net
        }
        
        if args.is_train:
            if not os.path.exists(args.graphs_dir):
                os.makedirs(args.graphs_dir)
            
            self.tensorboard_writer = SummaryWriter(logdir=args.graphs_dir, flush_secs=30)

            print('Writing graphs to {}'.format(args.graphs_dir))

    def preprocess_test_data(self, sampled_batch):
        """
        sampled_batch: (batch_size, img_height, img_width, channels)
        """
        args = self.args
        
        tgt_view = sampled_batch
        tgt_view = tgt_view.to(device).float()
        tgt_view *= 1./255.
        self.tgt_view = tgt_view*2.0 - 1.0
    
        #shape:  #scale, #batch, #chnls, h,w
        self.tgt_view_pyramid = scale_pyramid(self.tgt_view, args.num_scales)
        #shape:  #scale, #batch*#src_views, #chnls,h,w
        self.tgt_view_tile_pyramid = [
            self.tgt_view_pyramid[scale].repeat(args.num_source, 1, 1, 1)
            for scale in range(args.num_scales)
        ]
        
        self.src_views = None
        self.intrinsics = None
        self.src_views_concat = None
        self.src_views_pyramid = None
        self.multi_scale_intrinsices = None
 
    def iter_data_preparation(self, sampled_batch):
        args = self.args
        # sampled_batch: tgt_view, src_views, intrinsics
        
        # shape: batch, ch, h,w
        tgt_view = sampled_batch[0]
        
        # shape: batch, num_source*ch, h, w
        src_views = sampled_batch[1]
        
        # shape: batch, 3, 3
        intrinsics = sampled_batch[2]
        
        # The images here are integral (0-255)
        # shape: batch, 3, h, w
        self.tgt_view = tgt_view.to(device).float()
        self.tgt_view *= 1./255.
        self.tgt_view = self.tgt_view*2. - 1.
        
        self.src_views = src_views.to(device).float()
        self.src_views *= 1./255.
        self.src_views = self.src_views*2. - 1.
        #print(self.src_views, self.tgt_view)
        
        self.intrinsics = intrinsics.to(device).float()
        # shape: b*src_views,3,h,w
        self.src_views_concat = torch.cat([
            self.src_views[:, 3*s:3*(s + 1), :, :]
            for s in range(args.num_source)
        ], dim=0)
        

        #shape:  #scale, #batch, h,w, ch
        self.tgt_view_pyramid = scale_pyramid(self.tgt_view, args.num_scales)
                
        #shape:  #scale, #batch*#src_views, #chnls,h,w
        self.tgt_view_tile_pyramid = [
            self.tgt_view_pyramid[scale].repeat(args.num_source, 1, 1, 1)
            for scale in range(args.num_scales)
        ]

        #shape: scales, b*src_views, h, w, ch
        self.src_views_pyramid = scale_pyramid(self.src_views_concat,
                                               args.num_scales)

        # output multiple disparity prediction
        self.multi_scale_intrinsices = compute_multi_scale_intrinsics(
            self.intrinsics, args.num_scales)
        
    def spatial_normalize(self, disp):
        curr_c, _, curr_h, curr_w = list(disp.size())
        disp_mean = torch.mean(disp, dim=(0, 2, 3), keepdim=True)
        disp_exp = disp_mean.expand(disp.size())
        return disp/disp_exp
        
    def build_dispnet(self):
        args = self.args
        # shape: batch, channels, height, width
        self.dispnet_inputs = self.tgt_view
        
        # for multiple disparity predictions,
        # cat tgt_view and src_views along the batch dimension
        if args.is_train:
            for s in range(args.num_source):    #opt.num_source = 3 - 1 = 2
                self.dispnet_inputs = torch.cat((self.dispnet_inputs, self.src_views[:, 3*s : 3*(s + 1), :, :]), dim=0)
            # [12, 3, 128, 416] - bs*3, channels, height, width

        # shape: pyramid_scales, #batch+#batch*#src_views, h,w
        self.disparities = self.disp_net(self.dispnet_inputs)
        self.loss_disparities = [d.squeeze(1).unsqueeze(3) for d in self.disparities]
        
        """
        Length = 4
        disparities[0]: (12, 1, 128, 416)
        disparities[1]: (12, 1, 64, 208)
        disparities[2]: (12, 1, 32, 104)
        disparities[3]: (12, 1, 16, 52)
        """
        # shape: pyramid_scales, bs, h,w
        
        #self.depth = [self.spatial_normalize(disp) for disp in self.disparities]
        
        self.depth = [1.0/disp for disp in self.disparities]
        
        self.depth = [d.squeeze_(1) for d in self.depth]    #is this necessary? Yes, in the tf implementation it is done inside the compute_rigid_flow function
        
        self.loss_depth = [d.unsqueeze(3) for d in self.depth]
        
#         print(self.depth)
        """
        For training data:
        Length = 4
        depth[0]: (12, 128, 416)
        depth[1]: (12, 64, 208)
        depth[2]: (12, 32, 104)
        depth[3]: (12, 16, 52)
        i.e. (batch_size*num_imgs, height, width)
        """

    def build_posenet(self):
        self.posenet_inputs = torch.cat((self.tgt_view, self.src_views), dim=1)        
        self.poses = self.pose_net(self.posenet_inputs)
        # (batch_size, num_source, 6)
    
    def build_rigid_warp_flow(self):
        global n_iter
        # NOTE: this should be a python list,
        # since the sizes of different level of the pyramid are not same
        """
        Uses self.poses and self.depth, computed through build_posenet() and build_dispnet(), respectively
        """
#         import pickle
        
#         infile = open('/ceph/raunaks/depth2.pkl', 'rb')
#         self.depth = pickle.load(infile)
#         self.depth = [torch.tensor(d).squeeze(3) for d in self.depth]
#         print(self.depth[0].size())
        
#         infile = open('/ceph/raunaks/pose2.pkl', 'rb')
#         self.poses = pickle.load(infile)
#         self.poses = torch.tensor(self.poses)
#         print(self.poses.shape)
        
#         infile = open('/ceph/raunaks/intrin2.pkl', 'rb')
#         self.multi_scale_intrinsices = torch.tensor(pickle.load(infile))
#         print(self.multi_scale_intrinsices.shape)
        
        args = self.args
        self.fwd_rigid_flow_pyramid = []
        self.bwd_rigid_flow_pyramid = []

        #print(self.depth[0].shape)
        for scale in range(args.num_scales):    #num_scales is 4

            for src in range(args.num_source):  #num_source is 2
                # self.depth: (4, 12, _, _)
                # self.poses: (4, 2, 6)
                # self.multi_scale_intrinsices: (4, 4, 3, 3)
                                
                # (4, h, w, 2) for each particular scale
                fwd_rigid_flow = compute_rigid_flow( # Checks out
                    self.poses[:, src, :],
                    self.depth[scale][:args.batch_size, :, :], #the first disparity
                    self.multi_scale_intrinsices[:, scale, :, :], False)
        
                # (4, h, w, 2)
                bwd_rigid_flow = compute_rigid_flow(
                    self.poses[:, src, :],
                    self.depth[scale][args.batch_size * (
                        src + 1):args.batch_size * (src + 2), :, :],
                    self.multi_scale_intrinsices[:, scale, :, :], True)
                
                if not src:
                    fwd_rigid_flow_cat = fwd_rigid_flow
                    bwd_rigid_flow_cat = bwd_rigid_flow
                else:
                    fwd_rigid_flow_cat = torch.cat(
                        (fwd_rigid_flow_cat, fwd_rigid_flow), dim=0)
                    bwd_rigid_flow_cat = torch.cat(
                        (bwd_rigid_flow_cat, bwd_rigid_flow), dim=0)
            
            # After the inner loop runs: fwd_rigid_flow_cat - (b*src_imgs, h, w, 2)
            
            self.fwd_rigid_flow_pyramid.append(fwd_rigid_flow_cat)
            self.bwd_rigid_flow_pyramid.append(bwd_rigid_flow_cat)

        #After the outer loop runs: fwd_rigid_flow_pyramid: (scales, b*src_imgs, h, w, 2) like (4, 8, h, w, 2)
        
        self.fwd_rigid_warp_pyramid = [
            flow_warp(self.src_views_pyramid[scale],
                      self.fwd_rigid_flow_pyramid[scale])
            for scale in range(args.num_scales)
        ]
                
#         print(self.fwd_rigid_warp_pyramid[0].shape, self.fwd_rigid_warp_pyramid) - different
#         print(self.tmp_pyramid[0].shape, self.tmp_pyramid)
        
        self.bwd_rigid_warp_pyramid = [
            flow_warp(self.tgt_view_tile_pyramid[scale],
                      self.bwd_rigid_flow_pyramid[scale])
            for scale in range(args.num_scales)
        ]

        #print(len(self.fwd_rigid_warp_pyramid), " ", self.fwd_rigid_warp_pyramid[0].size())
        #fwd_rigid_warp_pyramid: (8,128,416,3), (8,64,208,3), (8,32,104,3), (8,16,52,3)
        
        if n_iter % 10000 == 0:
            for j in range(len(self.fwd_rigid_warp_pyramid)):
                x = self.fwd_rigid_warp_pyramid[j].permute(0, 3, 1, 2)
                x = (x - torch.min(x))/(torch.max(x)-torch.min(x))
                self.tensorboard_writer.add_images('fwd_rigid_warp_scale' + str(j), x, n_iter)
 
            for j in range(len(self.bwd_rigid_warp_pyramid)):
                x = self.fwd_rigid_warp_pyramid[j].permute(0, 3, 1, 2)
                x = (x - torch.min(x))/(torch.max(x)-torch.min(x))
                self.tensorboard_writer.add_images('bwd_rigid_warp_scale' + str(j), x, n_iter)

        self.fwd_rigid_error_pyramid = [
            image_similarity(args.simi_alpha,
                             self.tgt_view_tile_pyramid[scale],
                             self.fwd_rigid_warp_pyramid[scale])
            for scale in range(args.num_scales)
        ]
        self.bwd_rigid_error_pyramid = [
            image_similarity(args.simi_alpha, self.src_views_pyramid[scale],
                             self.bwd_rigid_warp_pyramid[scale])
            for scale in range(args.num_scales)
        ]
        
        if n_iter % 10000 == 0:
            self.fwd_rigid_error_scale=[]
            self.bwd_rigid_error_scale=[]
            #fwd_rigid_error_pyramid[0]: (8, 3, 128, 416)

            for j in range(len(self.fwd_rigid_error_pyramid)):
                tmp=torch.mean(self.fwd_rigid_error_pyramid[j].permute(0, 3, 1, 2), dim=1, keepdim=True)
                #tmp: (8, 1, 128, 416) in 1st iteration
                self.tensorboard_writer.add_images('fwd_rigid_error_scale' + str(j), tmp, n_iter)
                self.fwd_rigid_error_scale.append(tmp)

            for j in range(len(self.bwd_rigid_error_pyramid)):
                tmp=torch.mean(self.bwd_rigid_error_pyramid[j].permute(0, 3, 1, 2), dim=1, keepdim=True)
                self.tensorboard_writer.add_images('bwd_rigid_error_scale' + str(j), tmp, n_iter)
                self.bwd_rigid_error_scale.append(tmp)

    #####################################################################################################
    """
    def build_flownet(self):

        # output residual flow
        # TODO: non residual mode
        #   make input of the flowNet
        # cat along the color channels
        # shapes: #batch*#src_views, 3+3+3+2+1,h,w

        fwd_flownet_inputs = torch.cat(
            (self.tgt_view_tile_pyramid[0], self.src_views_pyramid[0],
             self.fwd_rigid_warp_pyramid[0], self.fwd_rigid_flow_pyramid[0],
             L2_norm(self.fwd_rigid_error_pyramid[0], dim=1)),
            dim=1)
        bwd_flownet_inputs = torch.cat(
            (self.src_views_pyramid[0], self.tgt_view_tile_pyramid[0],
             self.bwd_rigid_warp_pyramid[0], self.bwd_rigid_flow_pyramid[0],
             L2_norm(self.bwd_rigid_error_pyramid[0], dim=1)),
            dim=1)

        # shapes: # batch
        flownet_inputs = torch.cat((fwd_flownet_inputs, bwd_flownet_inputs),
                                   dim=0)

        # shape: (#batch*2, (3+3+3+2+1)*#src_views, h,w)
        self.resflow = self.flow_net(flownet_inputs)

    def build_full_warp_flow(self):
        # unnormalize the pyramid flow back to pixel metric
        resflow_scaling = []
        # for s in range(self.num_scales):
        #     batch_size, _, h, w = self.resflow[s].shape
        #     # create a scale factor matrix for pointwise multiplication
        #     # NOTE: flow channels x,y
        #     scale_factor = torch.tensor([w, h]).view(1, 2, 1,
        #                                              1).float().to(device)
        #     scale_factor = scale_factor.repeat(batch_size, 1, h, w)
        #     resflow_scaling.append(self.resflow[s] * scale_factor)

        # self.resflow = resflow_scaling

        self.fwd_full_flow_pyramid = [
            self.resflow[s][:self.batch_size * self.num_source,:,:,:] +
            self.fwd_rigid_flow_pyramid[s][:,:,:,:] for s in range(self.num_scales)
        ]
        self.bwd_full_flow_pyramid = [
            self.resflow[s][:self.batch_size * self.num_source,:,:,:] +
            self.bwd_rigid_flow_pyramid[s][:,:,:,:] for s in range(self.num_scales)
        ]

        self.fwd_full_warp_pyramid = [
            flow_warp(self.src_views_pyramid[s], self.fwd_full_flow_pyramid[s])
            for s in range(self.num_scales)
        ]
        self.bwd_full_warp_pyramid = [
            flow_warp(self.tgt_view_tile_pyramid[s],
                      self.bwd_full_flow_pyramid[s])
            for s in range(self.num_scales)
        ]

        self.fwd_full_error_pyramid = [
            image_similarity(self.simi_alpha, self.fwd_full_warp_pyramid[s],
                             self.tgt_view_tile_pyramid[s])
            for s in range(self.num_scales)
        ]
        self.bwd_full_error_pyramid = [
            image_similarity(self.simi_alpha, self.bwd_full_warp_pyramid[s],
                             self.src_views_pyramid[s])
            for s in range(self.num_scales)
        ]
    """
    
    def build_losses(self):
        """
        # NOTE: geometrical consistency
        if self.train_flow:
            bwd2fwd_flow_pyramid = [
                flow_warp(self.bwd_full_flow_pyramid[s],
                          self.fwd_full_flow_pyramid[s])
                for s in range(self.num_scales)
            ]
            fwd2bwd_flow_pyramid = [
                flow_warp(self.fwd_full_flow_pyramid[s],
                          self.bwd_full_flow_pyramid[s])
                for s in range(self.num_scales)
            ]

            fwd_flow_diff_pyramid = [
                torch.abs(bwd2fwd_flow_pyramid[s] +
                          self.fwd_full_flow_pyramid[s])
                for s in range(self.num_scales)
            ]
            bwd_flow_diff_pyramid = [
                torch.abs(fwd2bwd_flow_pyramid[s] +
                          self.bwd_full_flow_pyramid[s])
                for s in range(self.num_scales)
            ]

            fwd_consist_bound_pyramid = [
                self.geometric_consistency_beta * self.fwd_full_flow_pyramid[s]
                * 2**s for s in range(self.num_scales)
            ]
            bwd_consist_bound_pyramid = [
                self.geometric_consistency_beta * self.bwd_full_flow_pyramid[s]
                * 2**s for s in range(self.num_scales)
            ]
            # stop gradient at maximum opeartions
            fwd_consist_bound_pyramid = [
                torch.max(s,
                          self.geometric_consistency_alpha).clone().detach()
                for s in fwd_consist_bound_pyramid
            ]

            bwd_consist_bound_pyramid = [
                torch.max(s,
                          self.geometric_consistency_alpha).clone().detach()
                for s in bwd_consist_bound_pyramid
            ]

            fwd_mask_pyramid = [(fwd_flow_diff_pyramid[s] * 2**s <
                                 fwd_consist_bound_pyramid[s]).float()
                                for s in range(self.num_scales)]
            bwd_mask_pyramid = [(bwd_flow_diff_pyramid[s] * 2**s <
                                 bwd_consist_bound_pyramid[s]).float()
                                for s in range(self.num_scales)]
        """
        args = self.args

        self.loss_rigid_warp = 0
        self.loss_disp_smooth = 0
        
        if args.train_flow:
            self.loss_full_warp = 0
            self.loss_full_smooth = 0
            self.loss_geometric_consistency = 0

        for s in range(args.num_scales):

            self.loss_rigid_warp += args.loss_weight_rigid_warp *\
                args.num_source/2*(
                    torch.mean(self.fwd_rigid_error_pyramid[s]) +
                    torch.mean(self.bwd_rigid_error_pyramid[s]))

#             print(self.loss_disparities[0].size())
#             print(torch.cat((self.tgt_view_pyramid[3], self.src_views_pyramid[3]), dim=0).size())
            self.loss_disp_smooth += args.loss_weight_disparity_smooth/(2**s) *\
                smooth_loss(self.loss_depth[s], torch.cat(
                    (self.tgt_view_pyramid[s], self.src_views_pyramid[s]), dim=0))

            """
            if self.train_flow:
                self.loss_full_warp += self.loss_weight_full_warp * self.num_source / 2 * (
                    torch.sum(
                        torch.mean(self.fwd_full_error_pyramid[s], 1, True) *
                        fwd_mask_pyramid[s]) / torch.mean(fwd_mask_pyramid[s])
                    + torch.sum(
                        torch.mean(self.bwd_full_error_pyramid[s], 1, True) *
                        bwd_mask_pyramid[s]) / torch.mean(bwd_mask_pyramid[s]))

                self.loss_full_smooth += self.loss_weigtht_full_smooth/2**(s+1) *\
                    (flow_smooth_loss(
                        self.fwd_full_flow_pyramid[s], self.tgt_view_tile_pyramid[s]) +
                        flow_smooth_loss(self.bwd_full_flow_pyramid[s], self.src_views_pyramid[s]))

                self.loss_geometric_consistency += self.loss_weight_geometrical_consistency / 2 * (
                    torch.sum(
                        torch.mean(fwd_flow_diff_pyramid[s], 1, True) *
                        fwd_mask_pyramid[s]) / torch.mean(fwd_mask_pyramid[s])
                    + torch.sum(
                        torch.mean(bwd_flow_diff_pyramid[s], 1, True) *
                        bwd_mask_pyramid[s]) / torch.mean(bwd_mask_pyramid[s]))
            """
            
        self.loss_total = self.loss_rigid_warp + self.loss_disp_smooth
        
        """
        if self.train_flow:
            print('full warp: {} full_smooth: {}, geo_con:{}'.format(self.loss_full_warp,self.loss_full_smooth,self.loss_geometric_consistency))
            self.loss_total += self.loss_full_warp + \
                self.loss_full_smooth + self.loss_geometric_consistency
        """
        
    def training_inside_epoch(self):
        global n_iter
        args = self.args
        
        print("Length of train loader: {}".format(len(self.train_loader)))
        for i, sampled_batch in enumerate(self.train_loader):
            """
            Length of train_loader: num_sequences/4
            Length of sampled_batch: 3
            sampled_batch[i] : [batch_size, channels, height, width]
            """
            start = time.time()
            
            self.iter_data_preparation(sampled_batch)           
            
            self.build_dispnet()
            self.build_posenet()
            
            self.build_rigid_warp_flow()
            
            if args.train_flow:
                self.build_flownet()
                self.build_full_warp_flow()
            
            self.build_losses()

            """
            if torch.cuda.is_available(): 
                print(torch.cuda.get_device_name(0))
                print('Memory Usage:')
                print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
                print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
            """

            self.optimizer.zero_grad()
            self.loss_total.backward()
            self.optimizer.step()

            
            if n_iter % 100 == 0:
                print('Iteration: {} \t Rigid-warp: {:.4f} \t Disp-smooth: {:.6f}\tTime: {:.3f}'.format(n_iter, self.loss_rigid_warp.item(), self.loss_disp_smooth.item(), time.time() - start))

                self.tensorboard_writer.add_scalar('total_loss', self.loss_total.item(), n_iter)
                self.tensorboard_writer.add_scalar('rigid_warp_loss', self.loss_rigid_warp.item(), n_iter)
                self.tensorboard_writer.add_scalar('disp_smooth_loss', self.loss_disp_smooth.item(), n_iter)

            if n_iter % args.output_ckpt_iter == 0 and n_iter != 0:
                path = '{}/{}_{}'.format(args.ckpt_dir, 'flow' if args.train_flow else 'rigid_depth', str(n_iter)+'.pth')
                
                torch.save({
                    'iter': i,
                    'disp_net_state_dict': self.disp_net.state_dict(),
                    'pose_net_state_dict': self.pose_net.state_dict(),
                    'loss': self.loss_total
                }, path)
            
            n_iter += 1


    def train(self):
        global n_iter
        global device
        args = self.args
        # Sets mode of models to 'train'
        if not args.train_flow:
            self.pose_net.train()
            self.disp_net.train()
        
        print('Constructing dataset object...')
        self.train_set = SequenceFolder(
            root=args.data_dir,
            seed=args.seed,
            split='train',
            img_height=args.img_height,
            img_width=args.img_width,
            sequence_length=args.sequence_length)

        print('Constructing dataloader object...')
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set,
            shuffle=True,
            drop_last=True,
            num_workers=args.data_workers,
            batch_size=args.batch_size,
            pin_memory=True)

        optim_params = [{
            'params': v.parameters(),
            'lr': args.learning_rate
        } for v in self.nets.values()]

        
        self.optimizer = torch.optim.Adam(
            optim_params,
            betas=(args.momentum, args.beta),
            weight_decay=args.weight_decay)
        
        print('Starting training for {} epochs...'.format(args.epochs))
        for epoch in range(args.epochs):
            print('-------------------------------EPOCH {}---------------------------------'.format(epoch))
           
            self.training_inside_epoch()
            

    @torch.no_grad()
    def test_depth(self):
        args = self.args
        # Sets mode of models to 'eval'
        if not args.train_flow:
            self.pose_net.eval()
            self.disp_net.eval()

        print('Constructing test dataset object...')
        self.test_set = testSequenceFolder(
            root=args.test_dir,
            seed=args.seed,
            split='test',
            img_height=args.img_height,
            img_width=args.img_width,
            sequence_length=args.sequence_length)

        print('Constructing test dataloader object...')
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set,
            shuffle=False,
            drop_last=False,
            num_workers=args.data_workers,
            batch_size=args.batch_size,
            pin_memory=True)

        print("Length of test loader: {}".format(len(self.test_loader)))

        pred_all = []
        for i, sampled_batch in enumerate(self.test_loader):
            """
            Length of test_loader: number of sequences/4
            sampled_batch : [batch_size, channels, height, width]
            """

            start = time.time()
            
            self.preprocess_test_data(sampled_batch)
            self.build_dispnet()
            
            pred = self.depth[0]
            # pred: (batch_size, height, width)
            
            for b in range(sampled_batch.shape[0]):
                pred_all.append(pred[b, :, :].cpu().numpy())

        
        save_dir_path = args.outputs_dir + os.path.basename(args.ckpt_dir)
        
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        
        save_path = save_dir_path + "/rigid__" + str(args.ckpt_index) + '.npy'
        
        print("Saving depth predictions to {}".format(save_path))
        np.save(save_path, pred_all)
