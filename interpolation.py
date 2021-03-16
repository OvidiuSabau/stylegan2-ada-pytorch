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
    num_steps                  = 500,
    w_avg_samples              = 10000,
    initial_learning_rate      = 0.1,
    initial_noise_factor       = 0.05,
    lr_rampdown_length         = 0.25,
    lr_rampup_length           = 0.05,
    noise_ramp_length          = 0.75,
    regularize_noise_weight    = 1e5,
    verbose                    = False,
    device: torch.device
):


    assert identity.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(device) # type: ignore

    # Compute q stats.
    logprint(f'Computing W midpoint and stddev using {w_avg_samples} samples...')
    w_samples = np.random.RandomState(123).randn(w_avg_samples, G.num_ws)
    w_avg = np.mean(w_samples, axis=0, keepdims=True)     # [G.w_dim]
    w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

    # Setup noise inputs.
    noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

    # Load VGG16 feature detector.
    # url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
    # with dnnlib.util.open_url(url) as f:
    #     vgg16 = torch.jit.load(f).eval().to(device)

    seg_channel_dict = {
        'h': 0,
        'i': 1,
        'b': 2
    }

    zeroTensor = torch.zeros(1, requires_grad=False).to(device)
    epsTensor = torch.tensor([1e-6], requires_grad=False).to(device)

    vgg16 = torch.jit.load('vgg16.pt').eval().to(device)
    segNet = torch.load('segNet.pt').eval().to(device)

    for name, parameter in segNet.named_parameters():
        parameter.requires_grad = False

    segNet_mean = torch.from_numpy(np.load('train-mean.npy')).float().to(device)
    segNet_std = torch.from_numpy(np.load('train-std.npy')).float().to(device)

    def apply_seg_mask(
        x: torch.Tensor,

        channel: int
    ):
        x_norm = (x - segNet_mean) / segNet_std
        segmentation = segNet(x_norm)['out']
        maxs = torch.max(segmentation, dim=1)[0].unsqueeze(1)
        afterSubtraction = segmentation - maxs + epsTensor
        finalMask = torch.round(torch.max(zeroTensor, afterSubtraction) / epsTensor)

        return x * finalMask[:, channel]

    # Features for identity image.
    if identity.shape[2] > 256:
        identity = F.interpolate(identity.unsqueeze(0).to(torch.float32), size=(256, 256), mode='area')

    masked_identity_img = apply_seg_mask(identity, seg_channel_dict['i']).to('cuda').to(torch.float32)
    identity_features = vgg16(masked_identity_img, resize_images=False, return_lpips=True)

    # Features for hair image.
    if hair.shape[2] > 256:
        hair = F.interpolate(hair.unsqueeze(0).to(torch.float32), size=(256, 256), mode='area')

    masked_hair_img = apply_seg_mask(hair, seg_channel_dict['h']).to('cuda').to(torch.float32)
    hair_features = vgg16(masked_hair_img, resize_images=False, return_lpips=True)

    # Loading the projection of images to save time when debugging
    w_h = torch.from_numpy(np.load("projected_w_h.npz")['w'][0]).to('cuda')  #(18,512)
    w_p = torch.from_numpy(np.load("projected_w_p.npz")['w'][0]).to('cuda')  #(18,512)

    # w_h = project(G, hair, device=torch.device('cuda'))[-1][0]
    # w_p = project(G, identity, device=torch.device('cuda'))[-1][0]
    #
    # np.savez(f'projected_w_h.npz', w=w_h.cpu().numpy())  hair image is pic1.jpg and identity is pic2.jpg
    # np.savez(f'projected_w_p.npz', w=w_p.cpu().numpy())

    q_opt = torch.nn.Parameter(torch.randn(size=w_p.shape[1:], dtype=torch.float32, requires_grad=True, device=device))

    # list of all target ws through optimization
    w_out = torch.zeros([num_steps] + list(w_h.shape), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([q_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True
    print("starting the iterations..")
    for step in range(num_steps):
        print("iteration ", step)
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_p) * w_noise_scale
        w_t = w_p + q_opt.sigmoid() * (w_h - w_p)
        ws = w_t.unsqueeze(0) + w_noise


        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        target_image = G.synthesis(ws, noise_mode='const')
        target_image = (target_image + 1) * (255/2)
        if target_image.shape[2] > 256:
            target_image = F.interpolate(target_image, size=(256, 256), mode='area')

        # TODO: img -> masked_img

        target_image_norm = (target_image - segNet_mean) / segNet_std
        segmentation = segNet(target_image_norm)['out']
        maxs = torch.max(segmentation, dim=1)[0].unsqueeze(1)
        afterSubtraction = segmentation - maxs + epsTensor
        finalMasks = torch.round(torch.max(zeroTensor, afterSubtraction) / epsTensor)

        hair_target_image = target_image * finalMasks[:, seg_channel_dict['h']]
        identity_target_image = target_image * finalMasks[:, seg_channel_dict['i']]

        # print(hair_target_image.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0).shape)
        # PIL.Image.fromarray(masked_hair_img.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)).save('hair.png')
        # PIL.Image.fromarray(masked_identity_img.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0).astype(np.uint8)).save('identity.png')

        # print("running target features through vgg")
        # Features for synth images.
        target_hair_features = vgg16(hair_target_image, resize_images=False, return_lpips=True)
        target_identity_features = vgg16(identity_target_image, resize_images=False, return_lpips=True)

        # Compute loss
        hair_dist = (target_hair_features - hair_features).square().sum()
        identity_dist = (target_identity_features - identity_features).square().sum()
        dist = alpha1 * hair_dist + alpha2 * identity_dist

        # # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        loss = dist
        loss += reg_loss * regularize_noise_weight

        # Step
        print("optimizer steps")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')

        # Save projected W for each optimization step.
        w_out[step] = w_t.detach()

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out
 

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--hair', 'hair_fname', required=True, metavar='FILE')
@click.option('--identity', 'identity_fname', required=True, metavar='FILE')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', type=bool, default=True, show_default=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
def run_projection(
    network_pkl: str,
    hair_fname: str,
    identity_fname: str,
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

    # Load hair image.
    hair_pil = PIL.Image.open(hair_fname).convert('RGB')
    w, h = hair_pil.size
    s = min(w, h)
    hair_pil = hair_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    hair_pil = hair_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    hair_uint8 = np.array(hair_pil, dtype=np.uint8)

    identity_pil = PIL.Image.open(identity_fname).convert('RGB')
    w, h = identity_pil.size
    s = min(w, h)
    identity_pil = identity_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    identity_pil = identity_pil.resize((G.img_resolution, G.img_resolution), PIL.Image.LANCZOS)
    identity_uint8 = np.array(identity_pil, dtype=np.uint8)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = interpolation(
        G,
        hair=torch.tensor(hair_uint8.transpose([2, 0, 1]), device=device), # pylint: disable=not-callable
        identity=torch.tensor(identity_uint8.transpose([2, 0, 1]), device=device),
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
            video.append_data(np.concatenate([identity_uint8, hair_uint8, synth_image], axis=1))
        video.close()

    # Save final projected frame and W vector.
    hair_pil.save(f'{outdir}/hair.png')
    identity_pil.save(f'{outdir}/identity.png')
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255/2)
    synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
    PIL.Image.fromarray(synth_image, 'RGB').save(f'{outdir}/synth.png')
    np.savez(f'{outdir}/synth_w.npz', w=projected_w.unsqueeze(0).cpu().numpy())

#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
