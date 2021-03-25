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

seg_channel_dict = {
    'h': 0,
    'i': 1,
    'b': 2
}

female_hair_dict = {
    272: 75,
    1942: 66,
    2012: 19068,
    3708: 88,
    4501: 1169,
    4832: 928,
    5406: 148,
    6332: 18979,
    6352: 75,
    7534: 124,
    9441: 2852,
    9625: 95
}

female_id_dict = {
    0: 1169,
    1: 124,
    2: 148,
    3: 18979,
    4: 19068,
    5: 2852,
    6: 36,
    7: 66,
    8: 75,
    9: 88,
    10: 928,
    11: 95
}

male_hair_dict = {
    660: 1279,
    1691: 2079,
    1898: 546,
    2028: 2070,
    2599: 5348,
    4902: 168,
    5820: 5580,
    6107: 5911,
    6982: 1326,
    8245: 5771,
    9285: 1749,
    9725: 761
}

male_id_dict = {
    0: 1279,
    1: 1326,
    2: 168,
    3: 1749,
    4: 2070,
    5: 2079,
    6: 5348,
    7: 546,
    8: 5580,
    9: 5771,
    10: 5911,
    11: 761
}

def get_distances(
    orig_path: str,
    synth_path: str,
    male = True
):
    hair_feat = {}
    identity_feat = {}

    # Load networks
    vgg16 = torch.jit.load('vgg16.pt').eval().to('cuda')
    segNet = torch.load('segNet.pt').eval().to('cuda')
    segNet_mean = torch.from_numpy(np.load('train-mean.npy')).float().to('cuda')
    segNet_std = torch.from_numpy(np.load('train-std.npy')).float().to('cuda')

    def apply_seg_mask(
        x: torch.Tensor,
        channel: int
    ):
        x_norm = (x - segNet_mean) / segNet_std
        segmentation = segNet(x_norm)['out']
        mask = torch.argmax(segmentation, dim=1)
        mask = mask.squeeze().to('cuda')
        x = x.squeeze()
        mask[mask != channel] = 1000
        x = torch.where(mask != 1000, x, torch.tensor(255.0).to('cuda'))
        return x.unsqueeze(0)
    
    # Load orig img features
    for img_fname in os.listdir(orig_path):
        print(f'Loading {img_fname}...') 
        img_id = int(img_fname.split('.')[0])
        orig_pil = PIL.Image.open(f'{orig_path}/{img_fname}').convert('RGB')
        w, h = orig_pil.size
        s = min(w, h)
        orig_pil = orig_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        orig_pil = orig_pil.resize((256, 256), PIL.Image.LANCZOS)
        orig_uint8 = np.array(orig_pil, dtype=np.uint8)
        orig = torch.tensor(orig_uint8.transpose([2, 0, 1])).to('cuda')
        orig = orig.unsqueeze(0).to('cuda').to(torch.float32)
        
        if orig.shape[2] > 256:
            orig = F.interpolate(orig, size=(256, 256), mode='area')
        masked_hair_img = apply_seg_mask(orig, seg_channel_dict['h']).to('cuda').to(torch.float32)
        masked_hair_img = masked_hair_img.to(torch.uint8)
        hair_features = vgg16(masked_hair_img, resize_images=False, return_lpips=True)
        hair_feat[img_id] = hair_features

        masked_identity_img = apply_seg_mask(orig, seg_channel_dict['i']).to('cuda').to(torch.float32)
        masked_identity_img = masked_identity_img.to(torch.uint8)
        identity_features = vgg16(masked_identity_img, resize_images=False, return_lpips=True)
        identity_feat[img_id] = identity_features

    # Loop through synth images
    with open('dist_data_projections_18x512.csv', 'w') as f:
        hair_dists = []
        identity_dists = []
        dists = []
        for img_fname in os.listdir(synth_path):
            print(f'Evaluating {img_fname}...')
            
            #***PROJ IDS***
            img_id = int(img_fname.split('-')[0])
            hair_id = img_id
            identity_id = img_id

            '''
            ***SYNTH IDS***
            img_fname_spl = img_fname.split('_')
            identity_id = int(img_fname_spl[1][1:])
            hair_id = int(img_fname_spl[2][1:-4])

            #***STARGAN IDS***
            img_fname_spl = img_fname.split('_')
            stargan_hair_id = int(img_fname_spl[0])
            stargan_identity_id = int(img_fname_spl[1])
            if male:
                hair_id = male_hair_dict[stargan_hair_id]
                identity_id = male_id_dict[stargan_identity_id]
            else:
                hair_id = female_hair_dict[stargan_hair_id]
                identity_id = female_id_dict[stargan_identity_id]
            '''
            orig_hair_features = hair_feat[hair_id]
            orig_identity_features = identity_feat[identity_id]

            synth_pil = PIL.Image.open(f'{synth_path}/{img_fname}').convert('RGB')
            w, h = synth_pil.size
            s = min(w, h)
            synth_pil = synth_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            synth_pil = synth_pil.resize((256, 256), PIL.Image.LANCZOS)
            synth_uint8 = np.array(synth_pil, dtype=np.uint8)
            synth = torch.tensor(synth_uint8.transpose([2, 0, 1])).to('cuda')
            synth = synth.unsqueeze(0).to('cuda').to(torch.float32)

            if synth.shape[2] > 256:
                synth = F.interpolate(synth, size=(256, 256), mode='area')
            masked_hair_img = apply_seg_mask(synth, seg_channel_dict['h']).to('cuda').to(torch.float32)
            masked_hair_img = masked_hair_img.to(torch.uint8)
            hair_features = vgg16(masked_hair_img, resize_images=False, return_lpips=True)

            masked_identity_img = apply_seg_mask(synth, seg_channel_dict['i']).to('cuda').to(torch.float32)
            masked_identity_img = masked_identity_img.to(torch.uint8)
            identity_features = vgg16(masked_identity_img, resize_images=False, return_lpips=True)

            hair_dist = (hair_features - orig_hair_features).square().sum()
            identity_dist = (hair_features - orig_identity_features).square().sum()
            dist = hair_dist + identity_dist
            hair_dists.append(hair_dist)
            identity_dists.append(identity_dist)
            dists.append(dist)

            f.write(f'{hair_id},{identity_id},{hair_dist},{identity_dist},{dist}\n')
        '''
        hair_mean = np.mean(hair_dists.numpy())
        identity_mean = np.mean(identity_dists.numpy())
        dist_mean = np.mean(dists.numpy())
        f.write(f'-1,-1,{hair_mean},{identity_mean},{dist_mean}')
        '''
    print('Finished successfully')

if __name__ == '__main__':
    get_distances('fid_sets/orig', 'fid_sets/projections_18x512')
