from dataloader.QRCodeDataset import QRCodeDataset
import os
from os.path import join as pjoin
import numpy as np


if __name__ == "__main__":
    """
    拆出 background patch.
    """
    qr_code_dataset = QRCodeDataset(annotations_dir="./data/paper_qr_label_yolo",
                                    img_dir="./data/paper_qr",
                                    predefined_class_file="./data/predefined_classes.txt")

    for image_data in qr_code_dataset:
        print(image_data)