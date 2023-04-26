from pathlib import Path
#from util.detectQRCode import useYolo
# useYolo.WIEGHT_PATH = "./util/detectQRCode/best.pt"
# assert Path(useYolo.WIEGHT_PATH).is_file()
from util.makeRandomQR import randomQR
from util.qrCodeValidator import qrValidator
from util.smearSomthingOn import doSmearing
import cv2


if __name__ == "__main__":

    ground_qr = randomQR.get_random_qrcode()

    assert qrValidator.is_qrcode(ground_qr)

    points1 = doSmearing.generate_random_points_near_edges(image, 7)
    points2 = doSmearing.generate_random_points_near_edges(image, 9)
    points3 = doSmearing.generate_random_points_near_edges(image, 11)

    have_cur = doSmearing.draw_bezier_curve(image, [points1, points2, points3])

