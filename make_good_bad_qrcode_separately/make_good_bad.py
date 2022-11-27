from pathlib import Path
import imgaug.augmenters as iaa
import imageio
import cv2
import matplotlib.pyplot as plt


def make_good_bad(good_path, bad_path,ref_path):
    good_path = Path(good_path)
    bad_path = Path(bad_path)
    ref_path = Path(ref_path)
    assert good_path.exists()
    assert bad_path.exists()
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
                                fill_per_channel=0.5, RANDOM_RANGE=0.25)
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


if __name__ == "__main__":
    ref_path = "./../colab/data/perspective_qrCodes"
    good_path = "./../colab/data/good"
    bad_path = "./../colab/data/bad"
    # 從 ref_path 拿 圖片 並破壞 放置到 bad_path, 好的放 good_path
    make_good_bad(good_path, bad_path, ref_path)
