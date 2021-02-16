import glob
import numpy as np
import cv2
import os


# download dataset and unzip: https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view
rgb_images = glob.glob("CelebAMask-HQ/CelebA-HQ-img/*.jpg")
dict_imgs = {}

for file in rgb_images:
    name = file.split('/')[-1].split('.')[0]
    img = cv2.imread(file) # (1024, 1024, 3)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    dict_imgs[name] = img
    # np.save('CelebAMaskProcessed/{}.npy'.format(name), img)

np.savez_compressed('CelebAProcessedImages', **dict_imgs)


mask_names = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
dict_masks = {}

for i in range(15):
    mask_images = {}
    files = glob.iglob('CelebAMask-HQ/CelebAMask-HQ-mask-anno/{}/*.png'.format(i), recursive=True)
    for filename in files:
        if os.path.isfile(filename):
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (256, 256))
            details = filename.split('/')[-1].split('_', 1)
            name = int(details[0])
            type = details[1].split('.')[0]
            if name not in mask_images:
                mask_images[name] = {}
            mask_images[name][type] = img

    for img in mask_images:
        images = []
        for mask in mask_names:
            if mask in mask_images[img]:
                images.append(mask_images[img][mask])
            else:
                images.append(np.zeros((256, 256)))
        numpy_img = np.stack(images, axis=2)
        numpy_img = numpy_img.astype('bool')
        # np.save('CelebAMaskProcessedMasked/{}.npy'.format(img), numpy_img)
        dict_masks[str(img)] = numpy_img

np.savez_compressed('CelebAProcessedMaskedImages', **dict_masks)
