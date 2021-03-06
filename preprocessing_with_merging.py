import glob
import numpy as np
import cv2
import os
import gc


mask_names = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
img_shape = (1024, 1024)

print("Creating output folder FullyProcessedCelebA..")
path = os.getcwd() + '/FullyProcessedCelebA'
if not os.path.exists(path):
    os.makedirs(path)

# process all masks, these are split into 15 folders
for i in range(1):
    print("Processing masks folder {}..".format(i))
    mask_images = {}  # to store current folder's masks, all masks for one image are in one folder
    files = glob.glob('CelebAMask-HQ/CelebAMask-HQ-mask-anno/{}/*.png'.format(i))
    # collect all masks first
    for filename in files:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)  # (512, 512)
        img = cv2.resize(img, img_shape, interpolation=cv2.INTER_AREA)
        details = filename.split('/')[-1].split('_', 1)
        img_number = str(int(details[0])) # to remove zeros from the number (00023 -> 23)
        type = details[1].split('.')[0]
        if img_number not in mask_images:
            mask_images[img_number] = {}
        mask_images[img_number][type] = img.astype('bool')

    print("Folder {} masks collected, merging and saving masks..".format(i))
    # then process the masks
    for img_number in mask_images:
        # create identity mask
        mask_images[img_number]['identity'] = mask_images[img_number]['skin'].copy()
        # delete images with hats
        if 'hat' in mask_images[img_number]:
            continue
        # append ears, neck, necklace, eye glasses, cloth and earrings to the identity of person
        if 'neck' in mask_images[img_number]:
            mask_images[img_number]['identity'] = np.bitwise_or(mask_images[img_number]['identity'], mask_images[img_number]['neck'])
        if 'neck_l' in mask_images[img_number]:
            mask_images[img_number]['identity'] = np.bitwise_or(mask_images[img_number]['identity'], mask_images[img_number]['neck_l'])
        if 'l_ear' in mask_images[img_number]:
            mask_images[img_number]['identity'] = np.bitwise_or(mask_images[img_number]['identity'], mask_images[img_number]['l_ear'])
        if 'r_ear' in mask_images[img_number]:
            mask_images[img_number]['identity'] = np.bitwise_or(mask_images[img_number]['identity'], mask_images[img_number]['r_ear'])
        if 'eye_g' in mask_images[img_number]:
            mask_images[img_number]['identity'] = np.bitwise_or(mask_images[img_number]['identity'], mask_images[img_number]['eye_g'])
        if 'ear_r' in mask_images[img_number]:
            mask_images[img_number]['identity'] = np.bitwise_or(mask_images[img_number]['identity'], mask_images[img_number]['ear_r'])
        if 'cloth' in mask_images[img_number]:
            mask_images[img_number]['identity'] = np.bitwise_or(mask_images[img_number]['identity'], mask_images[img_number]['cloth'])
        # remove face pixels that are also hair
        if 'hair' in mask_images[img_number]:
            mask_images[img_number]['identity'][mask_images[img_number]['identity'] == mask_images[img_number]['hair']] = False
        else:
            mask_images[img_number]['hair'] = np.zeros(img_shape).astype('bool')

        # merge masks into one channel
        final_mask = np.ones(img_shape) * 2  # default is background with index 2
        final_mask[mask_images[img_number]['identity'] == 1] = 1  # identity is 1
        final_mask[mask_images[img_number]['hair'] == 1] = 0  # hair is 0

        # load RGB image, append it, transpose to (4, img_size) and save
        rgb_img = cv2.imread("CelebAMask-HQ/CelebA-HQ-img/{}.jpg".format(img_number))  # (1024, 1024, 3)
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        if rgb_img.shape[:2] != img_shape:
            rgb_img = cv2.resize(rgb_img, img_shape)

        full_image = np.dstack((rgb_img, final_mask))
        full_image = full_image.astype(np.uint8)
        np.save('FullyProcessedCelebA/{}.npy'.format(img_number), full_image.transpose(2, 0, 1))

    mask_images.clear()
    files.clear()
    gc.collect()
