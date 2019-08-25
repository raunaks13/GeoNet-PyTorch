import torch
import torch.nn.functional as F
import math
import numpy as np

device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')

def scale_pyramid(img, num_scales):
    # img: (b, ch, h, w)
    if img is None:
        return None
    else:

        # TODO: Shape of image is [channels, h, w]     
        b, ch, h, w = img.shape
        scaled_imgs = [img.permute(0,2,3,1)]
#         print(scaled_imgs[0])
        
        for i in range(num_scales - 1):
            ratio = 2 ** (i+1)
            nh = int(h/ratio)
            nw = int(w/ratio)
            
            scaled_img = torch.nn.functional.interpolate(img, size=(nh, nw), mode='area')
            scaled_img = scaled_img.permute(0, 2, 3, 1)
            
            scaled_imgs.append(scaled_img)        

        # scaled_imgs: (scales, b, h, w, ch)
        
    return scaled_imgs


def L2_norm(x, dim, keep_dims=True):
    curr_offset = 1e-10
    l2_norm = torch.norm(torch.abs(x) + curr_offset,
                         dim=dim, keepdim=keep_dims)
    return l2_norm

def DSSIM(x, y):
    
    avepooling2d = torch.nn.AvgPool2d(3, stride=1, padding=[1, 1])
    x = x.permute(0, 3, 1, 2)
    y = y.permute(0, 3, 1, 2)
    mu_x = avepooling2d(x)
    mu_y = avepooling2d(y)

    sigma_x = avepooling2d(x**2) - mu_x**2
    sigma_y = avepooling2d(y**2) - mu_y**2
    sigma_xy = avepooling2d(x*y) - mu_x*mu_y
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    # L_square = 255**2

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n/SSIM_d

    return torch.clamp((1 - SSIM.permute(0, 2,3,1))/2, 0, 1)

def gradient_x(img):    #checks out
    return img[:, :, :-1, :] - img[:, :, 1:, :]

def gradient_y(img):    #checks out
    return img[:, :-1, :, :] - img[:, 1:, :, :]

def compute_multi_scale_intrinsics(intrinsics, num_scales):

    batch_size = intrinsics.shape[0]
    multi_scale_intrinsices = []
    for s in range(num_scales):
        fx = intrinsics[:, 0, 0]/(2**s)
        fy = intrinsics[:, 1, 1]/(2**s)
        cx = intrinsics[:, 0, 2]/(2**s)
        cy = intrinsics[:, 1, 2]/(2**s)
        zeros = torch.zeros(batch_size).float().to(device)
        r1 = torch.stack([fx, zeros, cx], dim=1)  # shape: batch_size,3
        r2 = torch.stack([zeros, fy, cy], dim=1)  # shape: batch_size,3
        # shape: batch_size,3
        r3 = torch.tensor([0., 0., 1.]).float().view(
            1, 3).repeat(batch_size, 1).to(device)
        # concat along the spatial row dimension
        scale_instrinsics = torch.stack([r1, r2, r3], dim=1)
        multi_scale_intrinsices.append(
            scale_instrinsics)  # shape: num_scale,bs,3,3
    multi_scale_intrinsices = torch.stack(multi_scale_intrinsices, dim=1)
    return multi_scale_intrinsices

def euler2mat(z, y, x):
    global device
    # TODO: eular2mat
    '''
    input shapes of z,y,x all are: (#batch)
    '''
    batch_size = z.shape[0]

    _z = z.clamp(-np.pi, np.pi)
    _y = y.clamp(-np.pi, np.pi)
    _x = x.clamp(-np.pi, np.pi)

    ones = torch.ones(batch_size).float().to(device)
    zeros = torch.zeros(batch_size).float().to(device)

    cosz = torch.cos(z)
    sinz = torch.sin(z)
    # shape: (#batch,3)
    rotz_mat_r1 = torch.stack((cosz, -sinz, zeros), dim=1)
    rotz_mat_r2 = torch.stack((sinz, cosz, zeros), dim=1)
    rotz_mat_r3 = torch.stack((zeros, zeros, ones), dim=1)
    # shape: (#batch,3,3)
    rotz_mat = torch.stack((rotz_mat_r1, rotz_mat_r2, rotz_mat_r3), dim=1)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    roty_mat_r1 = torch.stack((cosy, zeros, siny), dim=1)
    roty_mat_r2 = torch.stack((zeros, ones, zeros), dim=1)
    roty_mat_r3 = torch.stack((-siny, zeros, cosy), dim=1)
    roty_mat = torch.stack((roty_mat_r1, roty_mat_r2, roty_mat_r3), dim=1)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    rotx_mat_r1 = torch.stack((ones, zeros, zeros), dim=1)
    rotx_mat_r2 = torch.stack((zeros, cosx, -sinx), dim=1)
    rotx_mat_r3 = torch.stack((zeros, sinx, cosx), dim=1)
    rotx_mat = torch.stack((rotx_mat_r1, rotx_mat_r2, rotx_mat_r3), dim=1)

    # shape: (#batch,3,3)
    rot_mat = torch.matmul(torch.matmul(rotx_mat, roty_mat), rotz_mat)
    
#     rot_mat = torch.matmul(rotz_mat, torch.matmul(roty_mat, rotx_mat))

    return rot_mat

def pixel2cam(depth, pixel_coords, intrinsics, is_homogeneous=True):
    global device
    
    """Transform coordinates in the pixel frame to the camera frame.
    Args:
        depth: depth maps -- [B, H, W]
        intrinsics_inv: intrinsics_inv matrix for each element of batch -- [B, 3, 3]
    Returns:
        array of (u,v,1) cam coordinates -- [B, 3, H, W]
    """
    
    b, h, w = depth.size()
    
    depth = depth.view(b, 1, -1)
    pixel_coords = pixel_coords.view(b, 3, -1)
    cam_coords = torch.matmul(torch.inverse(intrinsics), pixel_coords) * depth
    
    if is_homogeneous:
        ones = torch.ones(b, 1, h*w).float().to(device)
        cam_coords = torch.cat((cam_coords.to(device), ones), dim=1)
    
    cam_coords = cam_coords.view(b, -1, h, w)
    
    return cam_coords

def cam2pixel(cam_coords, proj):
    global device
    
    """Transforms coordinates in a camera frame to the pixel frame.

    Args:
    cam_coords: [batch, 4, height, width]
    proj: [batch, 4, 4]
    Returns:
    Pixel coordinates projected from the camera frame [batch, height, width, 2]
    """
    b, _, h, w = cam_coords.size()
    cam_coords = cam_coords.view(b, 4, h*w)
    unnormalized_pixel_coords = torch.matmul(proj, cam_coords)
    
    x_u = unnormalized_pixel_coords[:, :1, :]
    y_u = unnormalized_pixel_coords[:, 1:2, :]
    z_u = unnormalized_pixel_coords[:, 2:3, :]
    
    x_n = x_u / (z_u + 1e-10)
    y_n = y_u / (z_u + 1e-10)
        
    pixel_coords = torch.cat((x_n, y_n), dim=1)
    pixel_coords = pixel_coords.view(b, 2, h, w)
    
    return pixel_coords.permute(0, 2, 3, 1)

def pose_vec2mat(vec):
    global device
    # TODO:pose vec 2 mat
    # input shape of vec: (#batch, 6)
    # shape: (#batch,3)
    
    b, _ = vec.size()
    translation = vec[:, :3].unsqueeze(2)
    
    rx = vec[:, 3]
    ry = vec[:, 4]
    rz = vec[:, 5]
    
    rot_mat = euler2mat(rz, ry, rx)
    rot_mat = rot_mat.squeeze(1)
    
    filler = torch.tensor([0.,0.,0.,1.]).view(1, 4).repeat(b, 1, 1).float().to(device)
    
    transform_mat = torch.cat((rot_mat, translation), dim=2)
    transform_mat = torch.cat((transform_mat, filler), dim=1)
    
    return transform_mat

def meshgrid(batch, height, width, is_homogeneous=True):
    """Construct a 2D meshgrid.

    Args:
      batch: batch size
      height: height of the grid
      width: width of the grid
      is_homogeneous: whether to return in homogeneous coordinates
    
    Returns:
      x,y grid coordinates [batch, 2 (3 if homogeneous), height, width]
    """
    
    global device
    
    # (height, width)
    x_t = torch.matmul(
        torch.ones(height).view(height, 1).float().to(device),
        torch.linspace(-1, 1, width).view(1, width).to(device))
    
    # (height, width)
    y_t = torch.matmul(
        torch.linspace(-1, 1, height).view(height, 1).to(device),
        torch.ones(width).view(1, width).float().to(device))
    
    x_t = (x_t + 1) * 0.5 * (width-1)
    y_t = (y_t + 1) * 0.5 * (height-1)
        
    if is_homogeneous:
        ones = torch.ones_like(x_t).float().to(device)
        #ones = torch.ones(height, width).float().to(device)
        coords = torch.stack((x_t, y_t, ones), dim=0)  # shape: 3, h, w
    else:
        coords = torch.stack((x_t, y_t), dim=0)  # shape: 2, h, w
    
    coords = torch.unsqueeze(coords, 0).expand(batch, -1, height, width)

    return coords


def compute_rigid_flow(pose, depth, intrinsics, reverse_pose):
    global device
    '''Compute the rigid flow from src view to tgt view 

        input shapes:
            pose: (batch, 6)
            depth: (batch, h, w)
            intrinsics: (batch, 3, 3)
    '''
    b, h, w = depth.shape

    # shape: (batch, 4, 4)
    pose = pose_vec2mat(pose) # (b, 4, 4)
    if reverse_pose:
        pose = torch.inverse(pose) # (b, 4, 4)

    pixel_coords = meshgrid(b, h, w) # (batch, 3, h, w)

    tgt_pixel_coords = pixel_coords[:,:2,:,:].permute(0, 2, 3, 1)   # (batch, h, w, 2)
    cam_coords = pixel2cam(depth, pixel_coords, intrinsics) # (batch, 4, h, w)

    # Construct 4x4 intrinsics matrix
    filler = torch.tensor([0.,0.,0.,1.]).view(1, 4).repeat(b, 1, 1).to(device)
    intrinsics = torch.cat((intrinsics, torch.zeros((b, 3, 1)).float().to(device)), dim=2)
    intrinsics = torch.cat((intrinsics, filler), dim=1) # (batch, 4, 4)

    proj_tgt_cam_to_src_pixel = torch.matmul(intrinsics, pose)
    src_pixel_coords = cam2pixel(cam_coords, proj_tgt_cam_to_src_pixel)
    
    rigid_flow = src_pixel_coords - tgt_pixel_coords

    return rigid_flow


def flow_to_tgt_coords(src2tgt_flow):

    # shape: (#batch,2,h,w)
    batch_size, _,h,w = src2tgt_flow.shape
    
    # shape: (#batch,h,w,2)
    src2tgt_flow = src2tgt_flow.clone().permute(0,2,3,1)

    # shape: (#batch,h,w,2)
    src_coords = meshgrid(h, w, False).repeat(batch_size,1,1,1)

    tgt_coords = src_coords+src2tgt_flow

    normalizer = torch.tensor([(2./w),(2./h)]).repeat(batch_size,h,w,1).float().to(device)
    # shape: (#batch,h,w,2)
    tgt_coords = tgt_coords*normalizer-1

    # shape: (#batch,h,w,2)
    return tgt_coords


def flow_warp(src_img, flow):
    # src_img: (8, h, w, 3) 
    # flow: (8, h, w, 2)

    b, h, w, ch = src_img.size()
    tgt_pixel_coords = meshgrid(b, h, w, False).permute(0, 2, 3, 1) # (b, h, w, ch)
    src_pixel_coords = tgt_pixel_coords + flow
    
    output_img = bilinear_sampler(src_img, src_pixel_coords)

    return output_img


def bilinear_sampler(imgs, coords):
    global device
    """Construct a new image by bilinear sampling from the input image.

    Points falling outside the source image boundary have value 0.

    Args:
      imgs: source image to be sampled from [batch, height_s, width_s, channels]
      coords: coordinates of source pixels to sample from [batch, height_t,
        width_t, 2]. height_t/width_t correspond to the dimensions of the output
        image (don't need to be the same as height_s/width_s). The two channels
        correspond to x and y coordinates respectively.
    Returns:
      A new sampled image [batch, height_t, width_t, channels]
    """
    # imgs: (8, 128, 416, 3)
    # coords: (8, 128, 416, 2)
    
    def _repeat(x, n_repeats):
        global device
        rep = torch.ones(n_repeats).unsqueeze(0).float().to(device)
        x = torch.matmul(x.view(-1, 1), rep)
        return x.view(-1)
    
    coords_x = coords[:, :, :, 0].unsqueeze(3).float().to(device)
    coords_y = coords[:, :, :, 1].unsqueeze(3).float().to(device)
    
    inp_size = imgs.size()
    coord_size = coords.size()
    out_size = torch.tensor(coords.size())
    out_size[3] = imgs.size()[3]
    out_size = list(out_size)
    
    x0 = torch.floor(coords_x)
    x1 = x0 + 1
    y0 = torch.floor(coords_y)
    y1 = y0 + 1
    
    y_max = torch.tensor(imgs.size()[1] - 1).float()
    x_max = torch.tensor(imgs.size()[2] - 1).float()
    zero = torch.zeros([]).float()
    
    x0_safe = torch.clamp(x0, zero, x_max)
    y0_safe = torch.clamp(y0, zero, y_max)
    x1_safe = torch.clamp(x1, zero, x_max)
    y1_safe = torch.clamp(y1, zero, y_max)
    
    wt_x0 = x1_safe - coords_x
    wt_x1 = coords_x - x0_safe
    wt_y0 = y1_safe - coords_y
    wt_y1 = coords_y - y0_safe
    
    dim2 = torch.tensor(inp_size[2]).float().to(device)
    dim1 = torch.tensor(inp_size[2] * inp_size[1]).float().to(device)
    
    base_in = _repeat(torch.from_numpy(np.arange(coord_size[0])).float().to(device) * dim1, 
                      coord_size[1]*coord_size[2])
    
    base = torch.reshape(base_in, (coord_size[0], coord_size[1], coord_size[2], 1))
    
    base_y0 = base + y0_safe*dim2
    base_y1 = base + y1_safe*dim2
    
    idx00 = torch.reshape(x0_safe + base_y0, (-1,)).to(torch.int32).long()
    idx01 = torch.reshape(x0_safe + base_y1, (-1,)).to(torch.int32).long()
    idx10 = torch.reshape(x1_safe + base_y0, (-1,)).to(torch.int32).long()
    idx11 = torch.reshape(x1_safe + base_y1, (-1,)).to(torch.int32).long()

#     imgs_flat = torch.reshape(imgs, (-1, inp_size[3])).float()
    imgs_flat = imgs.contiguous().view(-1, inp_size[3]).float()

    im00 = torch.index_select(imgs_flat, 0, idx00).view(out_size)
    im01 = torch.index_select(imgs_flat, 0, idx01).view(out_size)
    im10 = torch.index_select(imgs_flat, 0, idx10).view(out_size)
    im11 = torch.index_select(imgs_flat, 0, idx11).view(out_size)
    
    
    w00 = wt_x0 * wt_y0
    w01 = wt_x0 * wt_y1
    w10 = wt_x1 * wt_y0
    w11 = wt_x1 * wt_y1

    output = (w00*im00) + (w01*im01) + (w10*im10) + (w11*im11)
    
    return output