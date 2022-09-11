from PIL import Image
from glob import glob
import numpy as np
import cv2
import random
import matplotlib.pyplot as plt


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


if __name__ == "__main__":

    # 要抓的圖片來源 root dir.
    IOU_estm_root = r'D:\Git\zjpj\data_clean\IoU_test_image'

    images = glob(IOU_estm_root + '/*.*')
    MAX_SIZE = (150, 150)
    ncols, nrows = 6, 6
    total_im = ncols * nrows
    number_of_picture = 3
    pick_list = []
    random.shuffle(images)
    for i in range(ncols * nrows * number_of_picture):
        pick_list.append(images[i])

    res = []
    for times in range(number_of_picture):
        vst = None
        for i in range(ncols):
            hst = None
            for j in range(nrows):
                image = Image.open(pick_list[(total_im * times) + i * ncols + j])
                image.thumbnail(MAX_SIZE)
                image = expand2square(image, (0, 0, 0))
                image = np.array(image)
                if hst is None:
                    hst = image
                else:
                    hst = np.hstack((hst, image))
            #
            if vst is None:
                vst = hst
            else:
                vst = np.vstack((vst, hst))
        #
        res.append(np.array(copy.deepcopy(vst)))

    for idx, image in enumerate(res):
        plt.imsave(f"./merge_{idx}.png", res[idx], dpi=300)