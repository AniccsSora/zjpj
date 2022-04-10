import os
from os.path import join as pjoin
import pathlib
import glob
import tqdm
from PIL import Image
import PIL
from pyzbar.pyzbar import decode as zb_decoder
from pyzxing import BarCodeReader as ZxingDecoder


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


    def detectQR_and_resave(self, src_dir, dist_dir):
        src_dir = self._rebuild_path(src_dir)
        dist_dir = self._rebuild_path(dist_dir)
        os.makedirs(dist_dir, exist_ok=True)

        _ = [_ for _ in src_dir.rglob('*.*')]
        _ = _[754:]
        zxing_decoder = ZxingDecoder()

        error_log = "error_detect_log.txt"
        if os.path.exists(error_log):
            os.remove(error_log)

        pbar = tqdm.tqdm(range(len(_)))
        for idx in pbar:
            ipath = _[idx]
            try:
                print(ipath)
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
            # if len(results2) != 0:
            #     print(results2)







if __name__ == "__main__":

    # QRCleaner("原始各種圖片來源Dir")
    cleaner = QRCleaner(r"C:\Users\Anicca\Desktop\QRcode")

    # 全部另存並轉成 RGB .png 格式，路徑於建構子內設定
    # "統一另存的資料夾位置"
    # cleaner._resave_image(r"./hello_test1")

    src_dir = r"./hello_test1"
    dist_dir = r"./valid_qrcode"
    # 將有效的圖片檔案，丟入 qrcode detector，有被detector的才算是可用資料。
    cleaner.detectQR_and_resave(src_dir, dist_dir)