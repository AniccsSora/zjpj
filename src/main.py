from pathlib import Path
from util.detectQRCode import useYolo
useYolo.WIEGHT_PATH = "./util/detectQRCode/best.pt"
assert Path(useYolo.WIEGHT_PATH).is_file()
import cv2


if __name__ == "__main__":

    imgs = [ _ for _ in Path("./data/raw").rglob("*.*")]

    for img in imgs:
        res_list = useYolo.detect_qr_field(img)
        print("detect:", len(res_list))
        for idx, _ in enumerate(res_list):
            print("")
            cv2.imshow(f"aaa {idx}", _)
        cv2.waitKey(0)
