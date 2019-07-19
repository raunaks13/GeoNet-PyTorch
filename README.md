# GeoNet - PyTorch version

**Note: This repository is a work in progress. Please review the details below. Contributions are welcome.**

This codebase recreates the Rigid Flow Reconstructor of the following paper:

> [GeoNet](https://arxiv.org/pdf/1803.02276.pdf): Unsupervised Learning of Dense Depth, Optical Flow, and Camera Pose (CVPR 2018)

> [Zhichao Yin](http://zhichaoyin.me/) and [Jianping Shi](http://shijianping.me/) (SenseTime Research)

The official TensorFlow implementation can be found [here](https://github.com/yzcjtr/GeoNet).

This repository contains the code corresponding to `train_rigid=True` in the original paper, i.e. for reconstructing rigid flow to train depth and pose.

## Requirements
This codebase was tested using PyTorch 1.0.0, CUDA 9.0, and Ubuntu 18.04 LTS.

You can install the following packages using `pip`:

```
torch >= 1.0.0
numpy
argparse
imageio
opencv-python
tensorboardX==1.7
```

## Data Preparation
Follow the instructions given in the [official repo](https://github.com/yzcjtr/GeoNet) for KITTI to download the training and testing datasets. 

For evaluation, you will have to insert a `gt_depth.npy` file in the folder `models/gt_data`
This ground truth depth file can be found [here](https://drive.google.com/open?id=1E9j6guYY2S_HXmUhkqw95IEmdevXyBqM). You are also welcome to use your own depth file - it's just a python list of depth maps for the KITTI testing dataset.

## Running

You can specify the parameters `is_train`, `ckpt_dir`, `data_dir`, `test_dir`, `graphs_dir`, `models_dir`, and `outputs_dir` in the bash file or through the command line.

Further information can be found in `GeoNet_main.py.`

To train DepthNet and PoseNet: `bash run.sh`

To test and evaluate depth: `bash eval_depth.sh`

## Results

Training has only been done for depth. For optimum performance, turn off `random_coloring` during data augmentation.

Loss and image warping visualizations can be visualized using `tensorboardX`, for which functionality has been implemented.

The current code achieves an absolute relative error of 0.27 on KITTI with a VGG backbone for the DepthNet and same hyperparameters as the original paper.

The original paper reports an absolute relative error of 0.164, so some hyperparameter tuning is still required.

## Contributing

There are a number of things which need to be added:

* Perfecting the result on depth (network tuning)
* Adding a function to test pose (since the pose network is already trained)
* Once these are done, the ResFlowNet for the NonRigid Motion Localizer can be implemented.

## Acknowledgements

The DepthNet and some utility functions were taken from [Clement Pinard's](https://github.com/ClementPinard/SfmLearner-Pytorch) implementation of SFMLearner in PyTorch. I have followed a similar file hierarchy as specified in [this](https://github.com/yijie0710/GeoNet_pytorch) unnoficial implementation.
