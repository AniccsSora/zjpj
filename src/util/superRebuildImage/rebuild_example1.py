from util.makeGoodAffineQR import makeAFQ
from util.smearSomthingOn import doSmearing
from util.qrCodeValidator import qrValidator
from PIL import Image, ImageFilter
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import coolRebuild
from util.qrCodeValidator import qrValidator


if __name__ == "__main__":
    clean_qr = [_ for _ in Path("./data/single_qr").rglob("*.*")][0:16]
    smear_qr = [_ for _ in Path("./data/smearing").rglob("*.*")][0:16]
    assert len(clean_qr) > 0 and len(smear_qr) > 0

    rebuild_res = []
    rebuild=None
    for i in range(len(clean_qr)):
        good = clean_qr[i]
        smearing = smear_qr[i]
        # Path > numpy
        good = Image.open(good)
        smearing = Image.open(smearing)
        #
        rebuild = coolRebuild.reality_rebuild(good, smearing, None)
        #rebuild_res.append(rebuild)
        #
        rebuild.save(f"./data/rebuild/{clean_qr[i].stem}.png")

    # plt.imshow(rebuild)
    # plt.show()

