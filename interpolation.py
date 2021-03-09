# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy

from projector import project

def interpolation(
    G,
    identity: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
    hair: torch.Tensor, # [C,H,W] and dynamic range [0,255], W & H must match G output resolution 
    *,
    alpha1                     = 1.0,
    alpha2                     = 1.0,
    num_steps                  = 2000,
    q_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):
'''
    input hair_img, person_img

    hair_img -> masked_hair_img
    person_img -> masked_person_img
    masked_person_img -> vgg -> person_features
    masked_hair_img -> vgg -> hair_features

    masked_imgs -> projection -> z_h, z_p -> mapping network -> w_h, w_p 
    disable grads for G
    define random Q requires_grad
    optimizer loop:
        w_t <- w_h + w_p
        G.synthesis(w_t) -> synth_image
        synth_img -> segmentation -> masked_hair_synth_img
        masked_hair_synth_img -> vgg -> masked_hair_synth_feat
        loss:
            (masked_hair_synth_feat - hair_features).square.sum for each
            define alphas

'''

    assert identity.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute q stats.
    logprint(f'Computing W midpoint and stddev using {q_avg_samples} samples...')
    q_samples = np.random.RandomState(123).randn(q_avg_samples, G.w_dim)  
    q_avg = np.mean(q_samples, axis=0, keepdims=True)     # [G.w_dim]
    q_std = (np.sum((q_samples - q_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    # url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    # with dnnlib.util.open_url(url) as f:
    #     vgg16 = torch.jit.load(f).eval().to(device)

    vgg16 = torch.jit.load('vgg16.pt').eval().to(device)

    # TODO: img -> masked_img
    masked_identity_img = identity
    masked_hair_img = hair

    # Features for identity image.
    masked_identity_img = identity.unsqueeze(0).to(device).to(torch.float32)
    if masked_identity_img.shape[2] > 256:
        masked_identity_img = F.interpolate(masked_identity_img, size=(256, 256), mode='area')
    identity_features = vgg16(masked_identity_img, resize_images=False, return_lpips=True)

    # Features for hair image.
    masked_hair_img = hair.unsqueeze(0).to(device).to(torch.float32)
    if masked_hair_img.shape[2] > 256:
        masked_hair_img = F.interpolate(masked_hair_img, size=(256, 256), mode='area')
    hair_features = vgg16(masked_hair_img, resize_images=False, return_lpips=True)

    # Projection of images to w
    w_h = project(G, hair)
    w_p = project(G, identity)

    q_opt = torch.tensor(q_avg, dtype=torch.float32, device=device, requires_grad=True) # pylint: disable=not-callable
    # list of all target ws through optimization
    w_out = torch.zeros([num_steps] + list(q_opt.shape[1:]), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([q_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        q_noise_scale = q_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        q_noise = torch.randn_like(q_opt) * q_noise_scale
        qs = (q_opt + q_noise)
        Q = torch.diag(qs)
        
        w_t = w_p + torch.matmul(Q, w_h - w_p) 

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        target_image = G.synthesis(w_t, noise_mode='const')
        target_image = (target_image + 1) * (255/2)
        if target_image.shape[2] > 256:
            target_image = F.interpolate(target_image, size=(256, 256), mode='area')

        # TODO: img -> masked_img
        hair_target_image = target_image
        identity_target_image = target_image

        # Features for synth images.
        target_hair_features = vgg16(hair_target_image, resize_images=False, return_lpips=True)
        target_identity_features = vgg16(identity_target_image, resize_images=False, return_lpips=True)

        # Compute loss
        hair_dist = (target_hair_features - hair_features).square().sum()
        identity_dist = (target_identity_features - identity_features).square().sum()
        dist = alpha1 * hair_dist + alpha2 * identity_dist

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_t

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
    network_pkl: str,
    target_fname: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int
):
    """Project given image to the latent space of pretrained network pickle.

    Examples:

    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(device) # type: ignore

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        num_steps=num_steps,
        device=device,
        verbose=True
    )
    print (f'Elapsed: {(perf_counter()-start_time):.1f} s')

    # Render debug output: optional video and projected image and W vector.
    os.makedirs(outdir, exist_ok=True)
    if save_video:
        video = imageio.get_writer(f'{outdir}/proj.mp4', mode='I', fps=10, codec='libx264', bitrate='16M')
        print (f'Saving optimization progress video "{outdir}/proj.mp4"')
        for projected_w in projected_w_steps:
            synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
            synth_image = (synth_image + 1) * (255/2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
            video.append_data(np.concatenate([target_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    target_pil.save(f'{outdir}/target.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/proj.png')
    np.savez(f'{outdir}/projected_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
