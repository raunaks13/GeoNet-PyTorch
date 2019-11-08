import argparse
import torch
from GeoNet_model import GeoNetModel
import os

def main():
    
    # When training/testing change --istrain, and the set of parameters specified below
    parser = argparse.ArgumentParser('description: GeoNet')
    
    parser.add_argument('--is_train', default=0, type=int,
                        help='whether to train or test')
    
    # Generally fixed parameters
    parser.add_argument('--train_flow', default=False, type=bool,
                        help='whether to train full flow or not')
    parser.add_argument('--sequence_length', default=3, type=int,
                        help='sequence length for each example')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='size of a sample batch')
    parser.add_argument('--epochs', default=30, type=int,
                        help='number of epochs to train on KITTI')
    parser.add_argument('--data_workers', default=8, type=int,
                        help='number of workers')
    parser.add_argument('--img_height', default=128, type=int,
                        help='height of KITTI image')
    parser.add_argument('--img_width', default=416, type=int,
                        help='width of KITTI image')
    parser.add_argument('--num_source', default=2, type=int,
                        help='number of source images')
    parser.add_argument('--num_scales', default=4, type=int,
                        help='number of scaling points')
    parser.add_argument('--seed', default=8964, type=int,
                        help='torch random seed')

    # Dataset directories
    parser.add_argument('--data_dir', default='/ceph/data/kitti_eigen_full/', 
                    help='directory of training dataset') 
    parser.add_argument('--test_dir', default='/ceph/data/kitti_raw/',
                        help='directory of testing dataset')
    
    # To edit during training
    parser.add_argument('--ckpt_dir', default='/ceph/raunaks/GeoNet-PyTorch/reconstruction/models/no-bn',
                        help='directory to save checkpoints')
    parser.add_argument('--graphs_dir', default='/ceph/raunaks/GeoNet-PyTorch/reconstruction/graphs/no-bn', 
                        help='directory to store tensorboard images and scalars')
    parser.add_argument('--output_ckpt_iter', default=5000, type=int,
                        help='interval to save checkpoints')
    
    # To edit during evaluation
    parser.add_argument('--outputs_dir', default='/ceph/raunaks/GeoNet-PyTorch/reconstruction/outputs/',
                        help='outer directory to save output depth models')
    parser.add_argument('--ckpt_index', default=150000, type=int,
                        help='the model index to consider while evaluating')
    
    # Training hyperparameters
    parser.add_argument('--simi_alpha', default=0.85, type=float,
                        help='alpha weight between SSIM and L1 in reconstruction loss')
    parser.add_argument('--loss_weight_rigid_warp', default=1.0, type=float,
                        help='weight for warping by rigid flow')
    parser.add_argument('--loss_weight_disparity_smooth', default=0.5, type=float,
                       help='weight for disp smoothness')
    parser.add_argument('--learning_rate', default=0.0002, type=float,
                        help='learning rate for Adam Optimizer')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum for Adam Optimizer')
    parser.add_argument('--beta', default=0.999, type=float,
                        help='beta for Adam Optimizer')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='weight decay for Adam Optimizer')
    
    """
    parser.add_argument('--geometric_consistency_alpha', default=3.0)
    parser.add_argument('--geometric_consistency_beta', default=0.05)
    parser.add_argument('--loss_weight_full_warp', default=1.0)
    parser.add_argument('--loss_weigtht_full_smooth', default=0.2)
    parser.add_argument('--loss_weight_geometrical_consistency', default=0.2)
    """
    
    args = parser.parse_args()

    import json
    with open('/ceph/raunaks/GeoNet-PyTorch/reconstruction/' + os.path.basename(args.graphs_dir) + '.txt', "w") as f:
        json.dump(args.__dict__, f, indent=2)

    if torch.cuda.is_available():
        print("CUDA available")
        device = torch.device('cuda')
    else:
        print("CUDA NOT available")
        exit()
        device = torch.device('cpu')

    geonet = GeoNetModel(args, device)

    if args.is_train:
        geonet.train()
    else:
        geonet.test_depth()

        
if __name__ == "__main__":
    main()
