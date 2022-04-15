import sys

sys.path.insert(1, "../dataloader")
from dataloader.QRCodeDataset import QRCodeDataset
from cv2 import cv2
import os
from os.path import join as pjoin
import random
import numpy as np
import matplotlib.pyplot as plt
import tqdm

predefined_class = ['qr-code', "bad-qr-code"]

def plot_one_box(x, image, color=None, label=None, line_thickness=None):
    """
    @param x: [x1, y1, x2, y2]。
    @param image: ndarray, 畫上去的目標。
    @param color: tuple, (R, G, B)。
    @param label: bbox 要寫上去的名稱。
    @param line_thickness: bbox 框線粗度。
    @return: 一張附有 bbox 的 ndarray。
    """
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (image.shape[0] + image.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))

    cv2.rectangle(image, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(image, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

    return image

#  函數：在一幅圖片對應位置上加上矩形框  image_name 圖片名稱不含後綴
def draw_box_on_image(image, colors, c_and_bboxes, shut_up=False):

    width, height = image.shape[1::-1]

    box_number = 0

    for c_b in c_and_bboxes:  # 例遍 txt文件得每一行
        box_class = predefined_class[int(c_b[0])]  # 拿名稱
        x, y, w, h = c_b[1:]  # 拿 box 參數
        x, y, w, h = float(x), float(y), float(w), float(h)
        if not shut_up:
            print(x, y, w, h, box_class)
            print("\t xywh:("
                  f"{round((x - w / 2) * width)},"
                  f"{round((y - h / 2) * height)},"
                  f"{round(w*width)},"
                  f"{round(h*height)})")

        x1 = round((x - w / 2) * width)
        y1 = round((y - h / 2) * height)
        x2 = round((x + w / 2) * width)
        y2 = round((y + h / 2) * height)


        image = plot_one_box([x1, y1, x2, y2], image, color=colors, label=box_class, line_thickness=None)
        box_number += 1

    return box_number, image


def get_32x32_boxes(img, c_and_bboxes, overlap=0.5, shut_up=False):
    """

    @param img: ndarray
    @param c_and_bboxes: [yolo_box_format, ...]， 'c' mean box.
    @param overlap: 0 < overlap <= 1
    @return boxes_32x32: [ bbox ...]
    """
    width, height = img.shape[1::-1]
    boxes_32x32 = []
    jump = overlap * 32

    for c_b in c_and_bboxes:
        x, y, w, h = c_b[1:]  # 拿 box 參數
        x, y, w, h = float(x), float(y), float(w), float(h)

        x1 = round((x - w / 2) * width)
        y1 = round((y - h / 2) * height)
        x2 = round((x + w / 2) * width)
        y2 = round((y + h / 2) * height)

        # 在 x1 2, y1 2 之間生成 sub boxes。

        sb = 0  # sub bbox counter
        _cc = 0.6 * 32  # 優化常數
        _x, _y = x1, y1  # left top in qr-coder region
        for yt in range(round((height*h-_cc)/jump)):
            for xt in range(round((width*w-_cc)/jump)):
                boxes_32x32.append((int(_x), int(_y), int(_x+32), int(_y+32)))
                _x += jump
                sb += 1
            # x軸 跑完
            _y += jump
            _x = x1

        if not shut_up:
            print(f"\t 共有 {sb} 個子框\n")

    return boxes_32x32

def yoloxywh2xyxy(yoloxywh, w, h):
    yolo_x, yolo_y, yolo_w, yolo_h = yoloxywh

    x1 = (w*yolo_x) - (w*yolo_w/2)
    y1 = (h*yolo_y) - (h*yolo_h/2)
    x2 = x1 + w * yolo_w
    y2 = y1 + h * yolo_h
    assert x1 >= 0
    assert y1 >= 0
    assert x2 >= 0
    assert y2 >= 0
    return int(x1), int(y1), int(x2), int(y2)

def resize_in_limit(image, limit):
    """
    @param image: ndarray 圖片
    @param limit: 最大邊長限制
    @return: 一個最長邊只有 limit 的 ndarray
    """
    w, h = image.shape[1::-1]
    if w > h:
        ratio = limit / w
    else:
        ratio = limit / h

    image = cv2.resize(image, (int(w*ratio), int(h*ratio)))

    return image

DIR_S = {
    "label_dir": "./data_clean/the_real593_label",  # 圖片集的 yolo format.txt 根目錄
    "image_dir": "./data_clean/the_real593",  # 圖片集
    "save_dir": "./data_clean/the_real593_patches",  # 32x32 的儲存資料夾
    "predefined": "./data_clean/classes.txt"  # class name 定義
}


if __name__ == "__main__":
    # 使用此程式前需要準備下二資料夾，內部已經存放好資料:
    # 1. QRCodes 資料夾 (多張圖片)
    # 2. 提供另外一個資料夾路徑，其內部只包含對應名稱的 label 資訊 (使用 yolo format)
    """ 全自動無腦 patch """
    resize_list = [1.0, 1.2, 1.3, 1.5, 1.8, 2.0]  # 縮放用
    # annotations_dir: 都是 txt(yolo註記法)
    # img_dir:圖片，與annotations_dir註記檔案名稱同名ㄟㄟㄟㄟㄟ。
    qr_code_dataset = QRCodeDataset(annotations_dir=DIR_S['label_dir'],
                                    img_dir=DIR_S["image_dir"],
                                    predefined_class_file=DIR_S["predefined"])
    qr_code_patches_save_dir = DIR_S["save_dir"]

    print("32 x 32 patches qrcode dataset 存檔路徑為:{}".format(qr_code_patches_save_dir))
    save_name_cnt = 1
    os.makedirs(qr_code_patches_save_dir, exist_ok=True)

    __VISUAL_DEMO__ = False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # range() 內第一個參數可控制從第幾張圖片開始。
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    pbar = tqdm.tqdm(range(0, len(qr_code_dataset)), smoothing=0.1, ncols=100)
    for data_idx in pbar:
        data = qr_code_dataset[data_idx]
        c_and_bboxes = data[0]  # class and bounding box
        image = data[1]  # 圖片本身
        image_name = data[2]
        pbar.set_description("processing: \"{}\"".format(image_name))
        MAX_SIDE_LIMIT = 666
        # 將最長邊限定在 MAX_SIDE_LIMIT
        image = resize_in_limit(image, limit=MAX_SIDE_LIMIT)

        sub_savedir = pjoin(qr_code_patches_save_dir, image_name)
        os.makedirs(sub_savedir)

        box_number, lined_image = draw_box_on_image(np.array(image), colors=(0, 0, 255),
                                                    c_and_bboxes=c_and_bboxes,
                                                    shut_up=True)

        # cv2.imshow("origin", image)
        if __VISUAL_DEMO__:
            cv2.imshow(f"[Demo] box_number = {box_number}", lined_image)

        boxes_32x32 = get_32x32_boxes(image, c_and_bboxes, overlap=1,
                                      shut_up=True)

        lb_image = np.array(image)
        if image.ndim == 2:
            lb_image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)

        if __VISUAL_DEMO__:
            for mini_box in boxes_32x32:
                lft, rbm = mini_box[0:2], mini_box[2:]  # left top, right bottom
                rc = [random.randint(0, 255) for _ in range(3)]  # random color
                cv2.rectangle(lb_image, lft, rbm, rc, 1, cv2.LINE_AA)
            cv2.imshow("[Demo] highlight ", lb_image)

        # 準備基礎畫布
        cnv = np.array(lined_image)
        if cnv.ndim == 2:
            cnv = cv2.cvtColor(cnv, cv2.COLOR_GRAY2BGR)

        _tmp = np.array(cnv)

        # 默認此次 bbox 框的都是 full qrcode (滿滿的 qrcode)
        jump_FLAG = False



        for idx, mini_box in enumerate(boxes_32x32):
            for resizeIdx, bbox_zoom in enumerate(resize_list):
                if bbox_zoom == 1.0:
                    lft, rbm = mini_box[0:2], mini_box[2:]  # left top, right bottom
                else:
                    lft, rbm = mini_box[0:2], mini_box[2:]
                    rbm = (int(mini_box[2:][0]+(32*bbox_zoom-32)), int(mini_box[2:][1]+(32*bbox_zoom-32)))

                if bbox_zoom == 1.0:
                    simg = image[lft[1]:rbm[1], lft[0]:rbm[0]]
                else:
                    base_size = 32
                    _add_range = int(base_size * bbox_zoom - base_size)
                    assert _add_range > 0
                    simg = image[lft[1]:rbm[1]+_add_range, lft[0]:rbm[0]+_add_range]
                # 名稱
                # sname = pjoin(sub_savedir, f"{save_name_cnt}.bmp")
                sname = pjoin(sub_savedir, f"{resizeIdx}_{str(idx).zfill(3)}.bmp")
                try:
                    cv2.imwrite(sname, simg)
                except Exception as e:
                    pass
                #save_name_cnt += 1
                # reset 畫布
                _tmp = np.array(cnv)
            # End of resize
        # end of one image.


