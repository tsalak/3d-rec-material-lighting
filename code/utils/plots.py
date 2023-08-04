import numpy as np
import torch
import torchvision
from PIL import Image

from utils import rend_util

def plot(write_idr, gamma, model, indices, model_outputs ,pose, rgb_gt, path, epoch, img_res, plot_nimgs, resolution):
    # arrange data to plot
    batch_size, num_samples, _ = rgb_gt.shape

    network_object_mask = model_outputs['network_object_mask']
    points = model_outputs['points'].reshape(batch_size, num_samples, 3)

    idr_rgb_eval = model_outputs['idr_rgb_values']
    idr_rgb_eval = idr_rgb_eval.reshape(batch_size, num_samples, 3)

    our_rgb_eval = model_outputs['our_rgb_values']
    our_rgb_eval = our_rgb_eval.reshape(batch_size, num_samples, 3)

    depth = torch.ones(batch_size * num_samples).cuda().float()
    if network_object_mask.sum() > 0:
        depth_valid = rend_util.get_depth(points, pose).reshape(-1)[network_object_mask]
        depth[network_object_mask] = depth_valid
        depth[~network_object_mask] = 0.98 * depth_valid.min()
    depth = depth.reshape(batch_size, num_samples, 1)

    normal = model_outputs['normal_values']
    normal = normal.reshape(batch_size, num_samples, 3)

    diffuse_albedo = model_outputs['our_diffuse_albedo_values']
    diffuse_albedo = diffuse_albedo.reshape(batch_size, num_samples, 3)
    
    # plot rendered images
    plot_images(write_idr, gamma, normal, idr_rgb_eval, diffuse_albedo, our_rgb_eval, rgb_gt, path, epoch, plot_nimgs, img_res)

    # plot depth maps
    plot_depth_maps(depth, path, epoch, plot_nimgs, img_res)

def plot_depth_maps(depth_maps, path, epoch, plot_nrow, img_res):
    depth_maps_plot = lin2img(depth_maps, img_res)

    tensor = torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                         scale_each=True,
                                         normalize=True,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/depth_{1}.png'.format(path, epoch))

def plot_images(write_idr, gamma, normal, idr_rgb_points, diffuse_albedo, our_rgb_points, ground_true, path, epoch, plot_nrow, img_res):
    ground_true = ground_true.cuda()

    tonemap_img = lambda x: torch.pow(x, 1./gamma)
    clip_img = lambda x: torch.clamp(x, min=0., max=1.)

    diffuse_albedo = clip_img(diffuse_albedo)
    # print('inside plot_images: ',  diffuse_albedo.min())
    our_rgb_points = clip_img(tonemap_img(our_rgb_points))
    ground_true = clip_img(tonemap_img(ground_true))
    normal = clip_img((normal + 1.) / 2.)

    if write_idr:
        idr_rgb_points = clip_img(tonemap_img(idr_rgb_points))
        output_vs_gt = torch.cat((normal, idr_rgb_points, diffuse_albedo, our_rgb_points, ground_true), dim=0)
    else:
        output_vs_gt = torch.cat((normal, diffuse_albedo, our_rgb_points, ground_true), dim=0)

    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/rendering_{1}.png'.format(path, epoch))

def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
