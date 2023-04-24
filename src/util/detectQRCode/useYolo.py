import torch
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import pathlib
from PIL import Image
import numpy as np
from typing import List, Union
import os


# yolo weight path
#wp = './exp2_補500張e100_arg_high/weights/best.pt'


# if nessarry, inject this path
WIEGHT_PATH = './best.pt'

def __get_xyxy(img:Union[pathlib.Path, str, np.ndarray], norm=False):
    assert os.path.exists(WIEGHT_PATH)
    model = torch.hub.load('ultralytics/yolov5', 'custom', WIEGHT_PATH, verbose=False)

    if isinstance(img, Path):
        w, h = cv2.imread(img.__str__()).shape[1::-1]
        result = model(img.__str__())
    elif isinstance(img, np.ndarray):
        w, h = img.shape[1::-1]
        result = model(img)
    elif isinstance(img, str):
        w, h = cv2.imread(img.__str__()).shape[1::-1]
        result = model(img)
    else:
        assert False  # 不支援的輸入型別


    xyxypc_res = []
    for xyxys in result.xyxy:
        for xyxy in xyxys:  # 一張圖片的判斷結果
            cpu_xyxy = xyxy.detach().cpu().numpy()
            x1, y1, x2, y2, p, cls_idx = cpu_xyxy
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if not norm:
                xyxypc_res.append((x1, y1, x2, y2, p, cls_idx))
            else:
                assert x1 / w <= 1
                assert y1 / h <= 1
                assert x2 / w <= 1
                assert y2 / h <= 1
                xyxypc_res.append((x1/w, y1/h, x2/w, y2/h, p, cls_idx))
    # p:機率, c:類別名
    return xyxypc_res

def __crop_image(image_path: Union[np.ndarray, str], coords):
    """
    裁剪圖像。
    Args:
        image_path (str): 要裁剪的圖像文件的路徑。
        coords (tuple): 一個包含左上角和右下角座標的元組 (x1, y1, x2, y2)。

    Returns:
        cv2 image (ndarray)

    """
    if isinstance(image_path, str):
        image = Image.open(image_path)
    elif isinstance(image_path, np.ndarray):
        image = Image.fromarray(image_path)
    else:
        assert False  # not support type

    cropped_image = image.crop(coords)
    cv2_image = cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR)
    return cv2_image


# 這是給外面call的
def detect_qr_field(img:Union[np.ndarray, str, pathlib.Path]) -> List[np.ndarray]:
    """
    @param img: 圖片
    @return: list: [qr_code]
    """

    if isinstance(img, np.ndarray):
        cv_im = img.copy()
    elif isinstance(img, pathlib.Path):
        cv_im = cv2.imread(img.__str__())
    elif isinstance(img, str):
        cv_im = cv2.imread(img)
    else:
        assert False  # 不支援的型別

    res = __get_xyxy(img)  # get_xyxy: 回傳 list，根據 bbox數量，裡面放置xyxypc資料
    single_qr_return = []
    for idx, res_single in enumerate(res):

        x1, y1, x2, y2, p, c = res_single
        # draw box on cv_im
        # cv2.rectangle(cv_im, (x1, y1), (x2, y2), (0, 255, 0), 3)
        single_qr = __crop_image(cv_im, (x1, y1, x2, y2))

        single_qr_return.append(single_qr)

    return single_qr_return

if __name__ == "__main__":
    """
    detect qrcode on image.
    """
    imgs = [_ for _ in Path("./data/raw").rglob("*.*")]

    for img in imgs:
        res_list = detect_qr_field(img)  # useYolo.detect_qr_field
        print("detect:", len(res_list))
        for idx, _ in enumerate(res_list):
            print("")
            #cv2.imshow(f"aaa {idx}", _)
            cv2.imwrite(f"./data/single_qr/{img.stem}_{idx}.png", cv2.cvtColor(_, cv2.COLOR_BGR2RGB))
        #cv2.waitKey(0)

