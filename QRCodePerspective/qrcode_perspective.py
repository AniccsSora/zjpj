import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import pyzbar
from PIL import Image
from pyzbar.pyzbar import ZBarSymbol
from pyzbar.pyzbar import decode as pyzbar_decoder
import pyzbar
import numpy as np


class DecodeResult:
    def __init__(self, decoded: pyzbar.pyzbar.Decoded):
        self.data = decoded.data
        self.type = decoded.type
        self.left_top = (decoded.rect.left, decoded.rect.top)
        self.w = decoded.rect.width
        self.h = decoded.rect.height
        self.right_buttom = (decoded.rect.left+decoded.rect.width, decoded.rect.top+decoded.rect.height)
        self.polygon = decoded.polygon
        self.orientation = decoded.orientation

class Custom_QRCodeDetector:
    def __init__(self):
        from pyzbar.pyzbar import decode as this_class_decode
        self.__detect = this_class_decode
        self._img = None
        self.decode_result = None

    def detect(self, img_path: str):
        img = Image.open(img_path)
        self._img = np.array(img)
        self._res = self.__detect(img)

        if len(self._res) == 1:
            self.decode_result = DecodeResult(self._res[0])
            self.__calc_perspective()
        #
        return self.decode_result

    def detect_and_get_norm_img(self, img_path: str):
        """
        取得放射變換後的 qrcode
        @param img_path:
        @return:
        """
        self.detect(img_path)
        return self._perpective_img

    def __calc_perspective(self):
        if self.decode_result is None:
            return
        orientation = self.decode_result.orientation
        w, h = self.decode_result.w, self.decode_result.h

        # counterclockwise-order 逆時針順
        p0, p1, p2, p3 = self.decode_result.polygon
        p0 = [p0.x, p0.y]
        p1 = [p1.x, p1.y]
        p2 = [p2.x, p2.y]
        p3 = [p3.x, p3.y]

        pts1 = np.float32([p0, p1, p2, p3])
        pts2 = np.float32([[0, 0], [0, h], [w, h], [w, 0]])

        # print("Debug: pts1:", pts1, end=" -> ")
        # print("pts2:", pts2)
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

        _size = int(max(w, h)*1.0)
        result = cv2.warpPerspective(self._img, matrix, (_size, _size))
        assert self._img is not None
        # cv2.imshow("raw image", self._img)
        # cv2.imshow("perspected", result)
        # cv2.waitKey(0)
        self._perpective_img = result

    def get_normorlized_qrcode(self):
        try:
            self._perpective_img
        except:
            return None
        return self._perpective_img


def perspective_qrcodes(src_path_root, dst_path_root):
    img_root = Path(src_path_root)
    SAVE_PATH = dst_path_root

    my_detector = Custom_QRCodeDetector()
    assert img_root.exists()
    imgs = [_ for _ in img_root.glob("*.*")]

    for img in imgs:
        fname = img.name
        new_PATH = Path(SAVE_PATH).joinpath(img.name)
        norm_img = my_detector.detect_and_get_norm_img(img.__str__())
        plt.imsave(new_PATH, norm_img)


if __name__ == "__main__":

    perspective_qrcodes("./../colab/data/qrCodes",  # 圖片來源 root
                        r"D:\Git\zjpj\colab\data\perspective_qrCodes")  # 存檔目的
    