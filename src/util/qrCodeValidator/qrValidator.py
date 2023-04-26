import cv2
from pyzbar import pyzbar
from PIL import Image
import numpy as np


def is_qrcode(image):
    if isinstance(image, str):
        image = cv2.imread(image)
    elif isinstance(image, np.ndarray):
        pass
    elif isinstance(image, Image.Image):
        image = np.asarray(image)
    else:
        raise ValueError("not support type:", type(image))

    if len(pyzbar.decode(image))> 0:
        return True
    else:
        return False

if __name__ == "__main__":
    # Load the image
    #image = cv2.imread('./data/output.png')
    image = cv2.imread('./data/0_0.png')


    print(is_qrcode(image))