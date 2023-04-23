from pathlib import Path
import imgaug.augmenters as iaa
import imageio
import cv2
import matplotlib.pyplot as plt
import os
import scipy.ndimage as ndimage
import imgaug.augmenters as iaa
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.distance import pdist
from skimage import data as sk_data
from skimage.color import gray2rgb
import matplotlib.pyplot as plt

# 定義遮蔽三角形區域的函數
def mask_triangle(img, random_state, parents, hooks):
    nb_triangles = 1
    alpha = 0.5
    height, width = img.shape[:2]
    for _ in range(nb_triangles):
        # 隨機選擇三個點，構成一個三角形
        points = np.random.randint(0, max(height, width), size=(3, 2))
        # 創建 Delaunay 三角剖分
        tri = Delaunay(points)
        # 計算每個像素到最近的三角形的距離
        distances = tri.transform[pdist(np.mgrid[0:height, 0:width].reshape(2, -1).T)]
        # 根據距離創建遮罩
        mask = np.zeros((height, width), dtype=np.float32)
        mask += alpha * np.exp(-distances**2)
        # 將遮罩應用到圖像上
        img = img * (1 - mask[..., np.newaxis])
    return img

# 定義遮蔽圓形區域的函數
def mask_circle(img, nb_circles = 1, circles_range=(0.05, 0.1)):
    """

    @param img:
                圖片本體
    @param nb_circles:
                幾個圓形
    @param circles_range:
                圓形的半徑比例，與最長邊做基準。
    @return:
                跟這個圖片一樣大的 但有圓形的 mask。
    """
    assert 0.0 < circles_range[0] < circles_range[1] < 1.0
    height, width = img.shape[:2]

    mask_total = np.zeros((height, width), dtype=np.uint8)

    mask_list = []
    # 根據距離創建遮罩

    for _ in range(nb_circles):
        mask = np.zeros((height, width), dtype=np.uint8)

        # Create a circle mask
        # 隨機選擇圓心和半徑
        # center = [x, y]
        center = np.random.randint(0, max(height, width), size=(2,))
        radius = np.random.randint(int(max(height, width)*circles_range[0]),
                                   int(max(height, width))*circles_range[1])
        # 計算每個像素到圓心的距離
        distances = np.sqrt(np.sum((np.mgrid[0:height, 0:width].T - center)**2, axis=2)).T

        # 走訪每個 mask 用 xy
        for y in range(height):
            for x in range(width):
                dis = np.sqrt((x-center[0])**2+(y-center[1])**2, dtype=np.float32)
                # 園內 highlight,
                mask[y, x] = 1 if dis < radius else 0
        # 存下這回合的 mask
        mask_list.append(mask)
        #
    # > 1 的是圓內
    # merge mask as z-axis
    for _ in mask_list:
        mask_total += _
    mask_total[mask_total >= 1] = 255
    # inverse
    mask_total = 255-mask_total

    return np.clip(mask_total, 0, 1)

def make_good_bad(good_path, bad_path,ref_path):
    good_path = Path(good_path)
    bad_path = Path(bad_path)
    ref_path = Path(ref_path)
    #assert good_path.exists()
    if not good_path.exists():
        os.mkdir(good_path)
    #assert bad_path.exists()
    if not bad_path.exists():
        os.mkdir(bad_path)
    assert ref_path.exists()
    position = ['left-top', #'uniform', 'normal', 'center',
                'left-bottom', #'left-center',  'center-top',
                'right-top', #'right-center', 'center-center', 'center-bottom',
                'right-bottom']
    import random
    random_use_potition = lambda : position[random.randint(0, len(position)-1)]
    random_size = lambda: random.uniform(0.15, 0.35)

    def get_seq():
        cutout_aug = iaa.Cutout(nb_iterations=2, position=random_use_potition(),
                                cval=(0, 255), fill_mode="constant", size=random_size(),
                                fill_per_channel=0.5)
        return iaa.Sequential([cutout_aug])

    for ref_img in ref_path.glob("*.*"):
        seq = get_seq()
        fname = ref_img.name
        ref_img = ref_img.__str__()
        image = imageio.v3.imread(ref_img)
        image_aug = seq(image=image)
        new_bad_PATH = bad_path.joinpath(fname)
        new_good_PATH = good_path.joinpath(fname)
        plt.imsave(new_good_PATH, image)
        plt.imsave(new_bad_PATH, image_aug)



""" 要修改 augimg 原始碼  Cutout > __init__ > _handle_position_parameter :
# 插入
import random
# RANDOM_RANGE = 0.2
assert 0.0 <= RANDOM_RANGE <= 1.0
min_random = random.uniform(0.0, RANDOM_RANGE) # 0.0
max_random = random.uniform(1.0-RANDOM_RANGE, 1.0) # 1.0
# random version mapping
mapping = {"top": min_random, "center": 0.5, "bottom": max_random, "left": min_random,
           "right": max_random}
"""
def make_image_have_circle(im):

    # Generate random mask
    mask = np.zeros_like(im)
    mask[128:192, 64:192] = 1

    # Apply the mask to the image
    masked_im = im * mask

    # Display the original image and the masked image
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(im, cmap='gray')
    ax1.set_title('Original Image')
    ax2.imshow(masked_im, cmap='gray')
    ax2.set_title('Masked Image')
    plt.show()

if __name__ == "__main__":
    ref_path = "./../colab/data/perspective_qrCodes"
    good_path = "./../colab/data/good1"
    bad_path = "./../colab/data/bad2"
    # 從 ref_path 拿 圖片 並破壞 放置到 bad_path, 好的放 good_path
    # make_good_bad(good_path, bad_path, ref_path)

    # 讀取範例圖片
    image = gray2rgb(sk_data.coins())

    have_circle_mask = mask_circle(image, nb_circles=5, circles_range=[0.05, 0.2])
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(image)
    ax1.set_title('Original Image')
    # expand [r,w] -> [r,w,1]
    have_circle_mask_1 = np.expand_dims(have_circle_mask, axis=-1)
    # [r,w, 1] -> [r,w, 3]
    have_circle_mask_3 = np.broadcast_to(have_circle_mask_1,
                                         (have_circle_mask_1.shape[0],
                                          have_circle_mask_1.shape[1], 3))
    ax2.imshow(have_circle_mask_3*image)
    ax2.set_title('Masked Image')
    plt.show()