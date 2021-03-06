from dataloader.QRCodeDataset import QRCodeDataset
import os
from os.path import join as pjoin
import numpy as np
from little_function import bbox_isOverlap, cutting_cube, analysis_yolo_row_data
import cv2
from tqdm import tqdm

if __name__ == "__main__":
    """
    拆出 background patch.
    """
    qr_code_dataset = QRCodeDataset(annotations_dir="./data_clean/the_real593_label",
                                    img_dir="./data_clean/the_real593",
                                    predefined_class_file="./data_clean/classes.txt")
    background_patch_saveDir = "./data_clean/new_background"

    # range 可以控制從哪一張圖片開始跑
    for image_idx in tqdm(range(0, len(qr_code_dataset))):
        image_data = qr_code_dataset[image_idx]
        cname_and_bboxes, image, fname = image_data
        # print(cname_and_bboxes, fname)

        img_w, img_h = image.shape[1::-1]

        qr_bboxes = analysis_yolo_row_data(cname_and_bboxes, w=img_w, h=img_h)

        # 有了預選的背景框框
        cube_gen = cutting_cube((img_w, img_h), 32, overlap=1.0)

        background_patches = []

        # 找到OK的 background
        for cube in cube_gen:
            for qr_bbox in qr_bboxes:
                if bbox_isOverlap(qr_bbox, cube):
                    break
            else:
                background_patches.append(cube)


        # 視覺化 背景框框
        # for bk in background_patches:
        #     lft, rbm = bk[0:2], bk[2:4]
        #     cv2.rectangle(image, lft, rbm, (255, 0, 0), 2)
        # cv2.imshow("hi", image)
        # cv2.waitKey(0)
        # continue

        save_path = pjoin(background_patch_saveDir, fname)
        os.makedirs(save_path)

        # 存下所有 background patches
        for bg_idx, background_patch in enumerate(background_patches):
            x1, y1, x2, y2 = background_patch
            bg_patch = image[y1:y2, x1:x2]
            cv2.imwrite(pjoin(save_path, f"{bg_idx+1}.bmp"), bg_patch)  # 檢查 channel.

