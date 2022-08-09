import os
from os.path import join as pjoin
import pathlib
import tqdm
from PIL import Image
import PIL
from pyzbar.pyzbar import decode as zb_decoder
import imagehash
import numpy as np
import cv2

class QRCleaner:
    def __init__(self, source_path):
        self.source_path = self._rebuild_path(source_path)

    def _rebuild_path(self, path, pure=False):
        """
        總是返回 fullpath
        關於 pure 的用意請參考 pathlib 文件.
        """
        if pathlib.PurePath(path).is_absolute() is False:
            p = pathlib.PurePath(pjoin(os.getcwd(), path))
        else:
            p = pathlib.PurePath(path)

        if not pure:
            p = pathlib.Path(p)

        return p

    def _resave_image(self, saving_dir):
        scan_dir = self.source_path  # 圖片來源
        saving_dir = self._rebuild_path(saving_dir)  # 另存目的地

        assert self.source_path.is_dir()  # 檢查來源是目錄
        os.makedirs(saving_dir, exist_ok=True)

        suffix_set = set()

        for i in scan_dir.rglob('*.*'):
            suffix_set.add(i.suffix)

        unread_images=[]
        save_idx = 1
        for suffix in suffix_set:
            pgen = scan_dir.rglob(f'*{suffix}')
            flist = [_ for _ in pgen]
            pbar = tqdm.tqdm(range(len(flist)), smoothing=0.1, ncols=100)
            pbar.set_description("processing: \"{}\"".format(suffix))
            for idx in pbar:
                try:
                    im = Image.open(flist[idx])
                    im = im.convert(mode="RGBA")
                    im = im.convert(mode="RGB")
                    im.save(saving_dir.joinpath(f'{save_idx}.png'))
                    save_idx = save_idx + 1
                except PIL.UnidentifiedImageError as e:
                    unread_images.append(flist[idx])

        with open("unconvert image.txt", mode='w', encoding="utf-8") as f:
            unread_images = [str(_)+'\n' for _ in unread_images]
            f.writelines(unread_images)
        if len(unread_images) != 0:
            print(f"請開啟 unconvert image.txt 確認尚未轉換之圖片。")

    def detectQR_and_resaveValidQR(self, src_dir, dist_dir):
        src_dir = self._rebuild_path(src_dir)
        dist_dir = self._rebuild_path(dist_dir)
        os.makedirs(dist_dir, exist_ok=True)

        _ = [_ for _ in src_dir.rglob('*.*')]

        error_log = "error_detect_log.txt"
        if os.path.exists(error_log):
            os.remove(error_log)

        pbar = tqdm.tqdm(range(len(_)))

        for idx in pbar:
            ipath = _[idx]
            try:
                pbar.set_description("{}:{}".format(idx, ipath.name))
                img = Image.open(ipath)
                results1 = zb_decoder(img)
            except Exception as e:
                with open(error_log, "a") as f:
                    print("error:", ipath)
                    print(f"{ipath}\n", file=f)
                continue

            #results2 = zxing_decoder.decode(ipath)
            if len(results1) != 0:
                img.save(dist_dir.joinpath(f"{idx}.png"))

    def detectQR_and_resaveValidQR_ver2(self, src_dir, dist_dir, dist_dir_failed):
        """
        將會另存 掃描失敗、掃描成功 之 qrcode 到另外的資料夾。
        @param src_dir: 圖片來源
        @param dist_dir: 掃描成功放置處
        @param dist_dir_failed: 掃描失敗放置處
        @return: 不回傳
        """

        src_dir = self._rebuild_path(src_dir)
        dist_dir = self._rebuild_path(dist_dir)
        dist_dir_failed = self._rebuild_path(dist_dir_failed)

        os.makedirs(dist_dir, exist_ok=True)
        os.makedirs(dist_dir_failed, exist_ok=True)

        _ = [_ for _ in src_dir.rglob('*.*')]

        error_log = "error_detect_log.txt"
        if os.path.exists(error_log):
            os.remove(error_log)

        pbar = tqdm.tqdm(range(0, len(_)))

        for idx in pbar:
            ipath = _[idx]
            try:
                pbar.set_description("{}:{}".format(idx, ipath.name))
                img = Image.open(ipath)
                results1 = zb_decoder(img)
            except Exception as e:
                with open(error_log, "a") as f:
                    print("error:", ipath)
                    print(f"{ipath}\n", file=f)
                continue

            #results2 = zxing_decoder.decode(ipath)
            if len(results1) != 0:
                img.save(dist_dir.joinpath(f"{idx}.png"))
            else:
                img.save(dist_dir_failed.joinpath(f"{idx}.png"))

    def check_duplicated(self, tar_dir, save_dir, exist_ok=False):
        src_dir = self._rebuild_path(tar_dir)
        save_dir = self._rebuild_path(save_dir)
        os.makedirs(save_dir, exist_ok=exist_ok)

        assert src_dir.is_dir()

        hashfunc = imagehash.phash
        image_hash_set = set()
        hash_dict = dict()

        save_idx = 0
        _ = [aa for aa in src_dir.rglob("*.*")]
        pbar = tqdm.tqdm(range(len(_)), smoothing=0.1, ncols=100)
        for idx in pbar:
            fpath = _[idx]
            pbar.set_description("processing: \"{}\"".format(fpath))
            hash = hashfunc(Image.open(fpath))
            hex_key = str(hash)
            if hex_key in image_hash_set:
                continue
            else:
                image_hash_set.add(hex_key)
                hash_dict.update({hex_key: fpath})
                Image.open(fpath).save(save_dir.joinpath(f"{save_idx}.png"))
                save_idx += 1
        print("共有 {} 張照片為不同".format(len(image_hash_set)))

    def qr_xywh(self, src_dir, xywh_dir):
        src_dir = self._rebuild_path(src_dir)
        xywh_dir = self._rebuild_path(xywh_dir)
        os.makedirs(xywh_dir, exist_ok=True)


        for impath in src_dir.rglob("*.*"):
            im = Image.open(impath)
            w, h = im.size
            tk = zb_decoder(im)
            re = self.parse_position(tk, w, h)
            save_name = xywh_dir.joinpath(f"{impath.stem}.txt")
            with open(save_name, 'w') as f:
                 for _ in re:
                     _ = str(["{:.6f}".format(__) for __ in _])  # 轉成小數後 6 位數
                     _ = _.replace('\'', '').replace('[', '').\
                           replace(']', '').replace(',', '')
                     write2file = _
                     write2file = f"0 {write2file}\n"
                     f.write(write2file)
            # write file End


    def parse_position(self, tk_list, w, h):
        res = []
        for tk in tk_list:
            x_min = tk.rect.left
            x_max = tk.rect.left + tk.rect.width
            y_min = tk.rect.top
            y_max = tk.rect.top + tk.rect.height
            x_center = ((x_min+x_max)/2) / w
            y_center = ((y_min+y_max)/2) / h
            yolo_w = (x_max-x_min) / w
            yolo_h = (y_max - y_min) / h
            res.append([x_center, y_center, yolo_w, yolo_h])
        return res
    #python labelImg.py D:\Git\zjpj\data_clean\non_duplicated  D:\Git\zjpj\data_clean\label_xywh

    def check_resize(self, img: np.ndarray, ALLOW_MAXSIZE=600):
        w, h = img.shape[1::-1]

        if not w < ALLOW_MAXSIZE or not h < ALLOW_MAXSIZE:
            w_ratio, h_ratio = ALLOW_MAXSIZE/w, ALLOW_MAXSIZE/h
            w_ratio if w_ratio < h_ratio else h_ratio
            return cv2.resize(img, (int(w*w_ratio), int(h*w_ratio)))
        return img

    def resize_all(self, tar, resave):
        tar = self._rebuild_path(tar)
        resave = self._rebuild_path(resave)
        assert tar.is_dir()
        os.makedirs(resave, exist_ok=True)

        _ = [_ for _ in tar.rglob('*.*')]

        error_log = "error_detect_log.txt"
        if os.path.exists(error_log):
            os.remove(error_log)

        pbar = tqdm.tqdm(range(0, len(_)))

        for idx in pbar:
            ipath = _[idx]
            img = cv2.imread(str(ipath))
            img = self.check_resize(img, ALLOW_MAXSIZE=600)
            img_rePth = resave.joinpath(f"{ipath.stem}.png")
            cv2.imwrite(str(img_rePth), img)

if __name__ == "__main__":

    # QRCleaner("原始各種圖片來源Dir")
    cleaner = QRCleaner(r"C:\Users\lab87\Desktop\QRcode")

    # 1. 全部另存並轉成 RGB .png 格式，路徑於建構子內設定
    # "統一另存的資料夾位置"
    # cleaner._resave_image(r"./hello_png")

    # src_dir = r"./valid_qrcode"  # 來源資料夾
    # dist_dir = r"./valid_qrcode_again"  # 存放合法 QRcode 資料夾
    # 2. 將有效的圖片檔案，丟入 qrcode detector，有被detector的才算是可用資料。
    # cleaner.detectQR_and_resaveValidQR(src_dir, dist_dir)

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
