from util.makeGoodAffineQR import makeAFQ
from util.smearSomthingOn import doSmearing
from util.qrCodeValidator import qrValidator
from util.superRebuildImage import coolRebuild
from PIL import Image, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random


if __name__ == "__main__":
    clean_qr = [_ for _ in Path("./data/single_qr").rglob("*.*")][0:16]
    smear_qr = [_ for _ in Path("./data/smearing").rglob("*.*")][0:16]
    # for other module
    rebuilds = [_ for _ in Path("./data/rebuild").rglob("*.*")][0:16]
    #
    pick_idx = random.randint(0, len(clean_qr)-1)
    aff = Image.open(clean_qr[pick_idx])  # good
    smearing = Image.open(smear_qr[pick_idx])  # bad

    # 重建
    rebuild = coolRebuild.reality_rebuild(aff, smearing)
    #
    # 設置子圖
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    # 繪製每個子圖

    axes[0].imshow(aff)
    axes[1].imshow(smearing)
    if True:
        blur_rebuild = rebuild.filter(ImageFilter.BLUR)
        axes[2].imshow(blur_rebuild)
    else:
        axes[2].imshow(rebuild)

    # 設置子圖標題
    axes[0].set_title('origin')
    axes[1].set_title('smearing')
    axes[2].set_title('rebuild')
    # off axis
    for ax in axes:
        ax.set_axis_off()
    # plt.tight_layout()
    # 顯示圖像
    plt.show()