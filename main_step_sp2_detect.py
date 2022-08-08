from useYolo import get_xyxy as detect_img
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
import glob
import os
import logging
import model as qrcnn_model
import torch
from misc.F import set_logger
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from misc.F import ensure_folder, timestamp
from pathlib import Path
from PIL import Image


LOG_SAVE_FOLDER = f"log_save/{timestamp()}"

ensure_folder(LOG_SAVE_FOLDER)

set_logger(path=LOG_SAVE_FOLDER)
logger = logging.getLogger("main_sp2.py")


def draw_bbox(bboxes, img, draw_p, show_p = True):
    """
    @param bboxes: 要繪製的 bbox 坐標組
    @param img: str path
    @param draw_p: float, 大於該機率才繪製圖案
    @return: 畫好 box 的 img
    """
    line_color = (0, 255, 0)  # BGR

    if isinstance(img, str):
        #img = cv2.imread(img, cv2.IMREAD_COLOR)
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
        img = np.array(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))


    for bbox in bboxes:
        x1, y1, x2, y2, p, _ = bbox
        if p < draw_p:
            continue
        cv2.rectangle(img, (x1, y1), (x2, y2), line_color, 3, cv2.LINE_AA)
        if show_p:
            cv2.putText(img, str(int(p*100)), (x1-10, y1-15), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 255), 3, cv2.LINE_AA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def box_coodinate_generater(xyxy, num, size=32, overlap=0.1):
    res_coondis = []


def gkern(l=32, sig=10):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


if __name__ == "__main__":
    # 使用者\.cache\torch\hub\ultralytics_yolov5_master\utils\general.py
    # def set_logging():
    #   內部插入
    #   ++ log.setLevel(logging.ERROR)
    #

    # 檢測 img path
    detection_root = r"./data_clean/the_real593"
    img_list = glob.glob(detection_root+'/*.*')
    _save_pth = Path(os.getcwd()).joinpath(LOG_SAVE_FOLDER)
    print("SAVE PATH:", _save_pth)
    bboxes_res = None
    bboxex_results = []
    for im in img_list:
        bboxes_res = detect_img(im)

        # draw box
        have_box_img = draw_bbox(bboxes_res, im, draw_p=0.5, show_p=False)

        bboxex_results.append(have_box_img)

    for idx, img in enumerate(bboxex_results):
        fn = Path(img_list[idx]).stem
        plt.imshow(bboxex_results[idx], alpha=1.0, cmap=plt.get_cmap('gray'))
        plt.savefig(_save_pth.joinpath(f"{fn}_save.png"))
        plt.cla()




