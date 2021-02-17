import glob
import numpy as np
import cv2
import os


# download dataset and unzip: https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view
rgb_images = glob.glob("CelebAMask-HQ/CelebA-HQ-img/*.jpg")
dict_imgs = {}
mask_names = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
dict_masks = {}

# first get all RGB images, they are all in one folder and have normal filenames
for file in rgb_images:
    img_number = file.split('/')[-1].split('.')[0]
    img = cv2.imread(file) # (1024, 1024, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    dict_imgs[img_number] = img
    # to save separately
    # np.save('CelebAMaskProcessed/{}.npy'.format(img_number), img)

# to save at once in .npz file
# np.savez_compressed('CelebAProcessedImages', **dict_imgs)

# then get all masks, but these are split into 15 folders
for i in range(15):
    mask_images = {} # to store current folder's masks, all masks for one image are in one folder
    files = glob.glob('CelebAMask-HQ/CelebAMask-HQ-mask-anno/{}/*.png'.format(i))
    # collect all masks first
    for filename in files:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        details = filename.split('/')[-1].split('_', 1)
        img_number = str(int(details[0])) # to remove zeros from the number (00023 -> 23)
        type = details[1].split('.')[0]
        if img_number not in mask_images:
            mask_images[img_number] = {}
        mask_images[img_number][type] = img

    # then merge masks together for each image and create the background
    for img_number in mask_images:
        all_masks = []
        for mask in mask_names:
            if mask in mask_images[img_number]:
                all_masks.append(mask_images[img_number][mask])
            else:
                all_masks.append(np.zeros((256, 256)))
        final_mask = np.stack(all_masks, axis=2).astype('bool')
        background_mask = np.logical_not(np.bitwise_or.reduce(final_mask, axis=2))
        final_mask = np.dstack((final_mask, background_mask))
        dict_masks[img_number] = final_mask
        # to save masks separately
        # np.save('CelebAMaskProcessedMasked/{}.npy'.format(img_number), final_mask)

# to save masks together in one .npz file
# np.savez_compressed('CelebAProcessedMaskedImages', **dict_masks)

# finally merge the RGB image with mask and save them
path = os.getcwd() + '/FullyProcessedCelebA'
if not os.path.exists(path):
    os.makedirs(path)

for img_number in dict_imgs:
    full_image = np.dstack((dict_imgs[img_number], dict_masks[img_number]))
    np.save('FullyProcessedCelebA/{}.npy'.format(img_number), full_image)
