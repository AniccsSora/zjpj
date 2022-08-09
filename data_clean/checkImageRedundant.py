import os
from os.path import join as pjoin
import pathlib
import tqdm
from PIL import Image
import PIL
from pyzbar.pyzbar import decode as zb_decoder
import imagehash
from QRcleaner import QRCleaner


if __name__ == "__main__":

    # QRCleaner("原始各種圖片來源Dir")
    cleaner = QRCleaner(r"C:\Users\Anicca\Desktop\QRcode")

    # 1. 全部另存並轉成 RGB .png 格式，路徑於建構子內設定
    # "統一另存的資料夾位置"
    # cleaner._resave_image(r"./valid_qrcode")

    src_dir = r"./valid_qrcode"  # 比對來源 A
    dist_dir = r"./valid_qrcode_success_detected"  # 存放合法 QRcode 資料夾
    dist_dir_failed = r"./qrcode_failed"  # 存放掃不到 QRcode 的資料夾
    # 2. 將有效的圖片檔案，丟入 qrcode detector，有被detector的才算是可用資料。
    # cleaner.detectQR_and_resaveValidQR_ver2(src_dir, dist_dir, dist_dir_failed)

    # 2.5 圖片縮放
    re_dir = "./qrcode_failed"
    cleaner.resize_all(tar=re_dir, resave='./happy_re')


    # 3. 檢查資料集內有無重複資料
    # tar_dir = "./new_set801"
    # sav_dir = "./non_duplicated"
    # cleaner.check_duplicated(tar_dir=tar_dir, save_dir=sav_dir, exist_ok=True)

    # 4. 利用 qr detector 來標記 src 的圖，label 存放在 xywh_dir
    # src_dir = "./non_duplicated"  # 圖片DIR
    # xywh_dir = "./label_xywh"  # label 的存放目的DIR
    # cleaner.qr_xywh(src_dir, xywh_dir)

    # 執行 labelImg 指令  python labelImg.py {圖片 DIR} {LABEL 存放區}
    # python labelImg.py D:\Git\zjpj\data_clean\non_duplicated  D:\Git\zjpj\data_clean\label_xywh
