from util.makeGoodAffineQR import makeAFQ
from util.smearSomthingOn import doSmearing
from util.qrCodeValidator import qrValidator
from PIL import Image, ImageFilter
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


def reality_rebuild(good:Image.Image, bad:Image.Image, power=None)->Image.Image:
    # do diff
    diff=None
    if power is None:
        diff = np.asarray(bad) - good
    else:
        diff = np.asarray(bad) - np.asarray(good) * (power/100)
        diff = diff.astype(np.uint8)
    # do trans
    diff_rgba = Image.fromarray(diff).convert('RGBA')
    #
    # Get the size of the image
    width, height = diff_rgba.size

    # Create a new image with transparency
    have_aplha_diff = Image.new('RGBA', (width, height), (0, 0, 0, 0))

    # Copy the original image's pixels to the new image, replacing black pixels with transparency
    for x in range(width):
        for y in range(height):
            r, g, b, a = diff_rgba.getpixel((x, y))
            if r == 0 and g == 0 and b == 0:
                a = 0
            have_aplha_diff.putpixel((x, y), (r, g, b, a))
    #
    # algorithm session, processing alpha image do somthing...
    # diff processing
    _ = have_aplha_diff.filter(ImageFilter.GaussianBlur)
    _ = _.filter(ImageFilter.SMOOTH_MORE)
    #_ = _.filter(ImageFilter.SMOOTH_MORE)
    #_ = _.filter(ImageFilter.SMOOTH_MORE)
    _ = _.filter(ImageFilter.GaussianBlur)
    # _ = _.filter(ImageFilter.SMOOTH_MORE)
    # _ = _.filter(ImageFilter.SMOOTH_MORE)
    # _ = _.filter(ImageFilter.SMOOTH_MORE)
    # _ = _.filter(ImageFilter.GaussianBlur)
    #
    #  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    # copy to processing
    rebuild = good.copy()  # 最後結果
    rebuild_data = rebuild.load()  # 取數值用
    fking_little_bit_diff_rgba = _.load()  # 加料用 取數值用
    # Add the images together, pixel by pixel
    for y in range(good.size[1]):
        for x in range(good.size[0]):
            r1, g1, b1 = rebuild_data[x, y]
            r2, g2, b2, a2 = fking_little_bit_diff_rgba[x, y]
            if a2 > 0:
                r = r1 + r2
                g = g1 + g2
                b = b1 + b2
            else:
                r = r1
                g = g1
                b = b1

            rebuild.putpixel((x, y), (r, g, b))
    return rebuild
    pass


if __name__ == "__main__":

    good, aff = makeAFQ.get_good_affine_qr()
    #
    points = doSmearing.gen_curves_points_list(
        ref_image=aff,
        n=3,
        complixty=(3, 19)
    )
    # smearing: checking is good seamring.
    while True:
        smearing = doSmearing.draw_bezier_curve(aff, points, num_segments=max(aff.size) * 100)
        if qrValidator.is_qrcode(smearing) == False:
            # good smearing
            break
    # 重建
    rebuild = reality_rebuild(aff, smearing)
    #
    # 設置子圖
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))

    # 繪製每個子圖
    axes[0, 0].imshow(good)
    axes[0, 1].imshow(aff)
    axes[1, 0].imshow(smearing)
    blur_rebuild = rebuild.filter(ImageFilter.GaussianBlur)
    axes[1, 1].imshow(blur_rebuild)

    # 設置子圖標題
    axes[0, 0].set_title('good')
    axes[0, 1].set_title('affine')
    axes[1, 0].set_title('smearing')
    axes[1, 1].set_title('rebuild')
    # off axis
    for ax in  axes.flatten():
        ax.set_axis_off()
    #plt.tight_layout()
    # 顯示圖像
    plt.show()