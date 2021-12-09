import sys

sys.path.insert(1, "../dataloader")
from dataloader.QRCodeDataset import QRCodeDataset
from cv2 import cv2
import os
from os.path import join as pjoin
import random
import numpy as np

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
def draw_box_on_image(image, colors, c_and_bboxes):

    width, height = image.shape[1::-1]

    box_number = 0

    for c_b in c_and_bboxes:  # 例遍 txt文件得每一行
        box_class = predefined_class[int(c_b[0])]  # 拿名稱
        x, y, w, h = c_b[1:]  # 拿 box 參數
        x, y, w, h = float(x), float(y), float(w), float(h)
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


def get_32x32_boxes(img, c_and_bboxes, overlap=0.5):
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

        print(f"\t 共有 {sb} 個子框\n")

    return boxes_32x32


if __name__ == "__main__":
    """ 將 QRCode 切成 32 x 32，這邊特別處理 bbox 內部的 QRCode 
        因為當 QRCode 傾斜 45度時會造成 四邊的 patch 框的不屬於 QRCode """
    # annotations_dir: 都是 txt(yolo註記法)
    # img_dir:圖片，與annotations_dir註記檔案名稱同名ㄟㄟㄟㄟㄟ。
    qr_code_dataset = QRCodeDataset(annotations_dir="./data/paper_qr_label_yolo",
                                    img_dir="./data/paper_qr",
                                    predefined_class_file="./data/predefined_classes.txt")
    qr_code_patches_save_dir = "./data/pathes_of_qrcode_32x32"
    save_name_cnt = 1
    os.makedirs(qr_code_patches_save_dir, exist_ok=True)

    __VISUAL_DEMO__ = False

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # range() 內第一個參數可控制從第幾張圖片開始。
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for data_idx in range(35, len(qr_code_dataset)):
        data = qr_code_dataset[data_idx]
        c_and_bboxes = data[0]  # class and bounding box
        image = data[1]  # 圖片本身
        image_name = data[2]

        sub_savedir = pjoin(qr_code_patches_save_dir, image_name)
        os.makedirs(sub_savedir)

        box_number, lined_image = draw_box_on_image(np.array(image), colors=(0, 0, 255), c_and_bboxes=c_and_bboxes)

        # cv2.imshow("origin", image)
        if __VISUAL_DEMO__:
            cv2.imshow(f"[Demo] box_number = {box_number}", lined_image)

        boxes_32x32 = get_32x32_boxes(image, c_and_bboxes, overlap=1)

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

        # bbox的放大率，預設為 1.0
        bbox_zoom = 1.0

        for idx, mini_box in enumerate(boxes_32x32):
            if bbox_zoom == 1.0:
                lft, rbm = mini_box[0:2], mini_box[2:]  # left top, right bottom
            else:
                lft, rbm = mini_box[0:2], mini_box[2:]
                rbm = (int(mini_box[2:][0]+(32*bbox_zoom-32)), int(mini_box[2:][1]+(32*bbox_zoom-32)))
            cv2.rectangle(_tmp, lft, rbm, (0, 255, 0), 1, cv2.LINE_AA)
            if not jump_FLAG:
                cv2.imshow(f"({idx+1}/{len(boxes_32x32)}) O or X ", _tmp)
                print(lft, rbm)

            while True:
                if not jump_FLAG:
                    k = cv2.waitKey(0) & 0xFF
                if (k == ord('o') or k == ord('O')) or jump_FLAG:
                    print(f"{idx+1} 按下了 o")
                    if bbox_zoom == 1.0:
                        simg = image[lft[1]:rbm[1], lft[0]:rbm[0]]
                    else:
                        base_size = 32
                        _add_range = int(base_size * bbox_zoom - base_size)
                        assert _add_range > 0
                        simg = image[lft[1]:rbm[1]+_add_range, lft[0]:rbm[0]+_add_range]
                    sname = pjoin(sub_savedir, f"{save_name_cnt}.bmp")
                    cv2.imwrite(sname, simg)
                    cv2.destroyAllWindows()
                    save_name_cnt += 1
                    break
                elif k == ord('x') or k == ord('X'):
                    print(f"{idx+1} 按下了 x")
                    cv2.destroyAllWindows()
                    break
                # 跳過這這張圖片，並默認全部都是 qrcode
                elif k == ord('a') or k == ord('A'):
                    print(f"{idx + 1} 按下了 A")
                    print("默認全部均是 qrcode patch")
                    jump_FLAG = True
                elif k == ord('r') or k == ord('R'):
                    print(f"{idx + 1} 按下了 r")
                    print("重新採樣 這張的 resolution")
                    new_bbox_zoom = float(input("框框要變大幾倍?:"))
                    print(f"bbox 放大率從 {bbox_zoom} --> {new_bbox_zoom}")
                    bbox_zoom = new_bbox_zoom
                    continue
                else:
                    continue

            # reset 畫布
            _tmp = np.array(cnv)
        # end of one image.


