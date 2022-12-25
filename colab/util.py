import numpy as np
import cv2
import glob
from pathlib import Path
import os
import shutil
import torch
from pathlib import Path

def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print(f"在 {os.path.curdir} 下 建立...")
        print("Creating folder {}".format(path))
        os.makedirs(path)

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's
    validation loss is less than the previous least less, then save the
    model state.
    """

    def __init__(
            self, save_root, best_valid_loss=float('inf')
    ):
        self.save_path = save_root
        ensure_dir(self.save_path)
        self.best_valid_loss = best_valid_loss

    def __call__(
            self, current_valid_loss, net, current_epoch,
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"Saving best model for epoch: {current_epoch}\n")
            _file_name = Path(self.save_path).joinpath("best.pt")
            torch.save(net.state_dict(), _file_name)
            # for colab special command
            # !cp /content/$_file_name /content/gdrive/MyDrive/ColabNotebooks/$_file_name
            return True
        else:
            return False

def pad_2_square(img: np.ndarray, export_size: int):
    """
    將一張圖 padding 成置中正方形。
    @param img: 輸入ndarray。
    @param export_size: 要輸出的邊長。
    @return res: 正方形的 ndarray.
    """

    is_3channel = True if img.ndim == 3 else False

    res = img.copy()

    w, h = res.shape[1::-1]
    if w == h:
        return cv2.resize(res, (export_size, export_size))

    max_side = max(w, h)
    min_side = min(w, h)

    if h > w:
        res = res.T
    align = abs(max_side-min_side)//2

    if abs(max_side-min_side) % 2 == 0:
        if is_3channel:
            pad_width = ((align, align), (0, 0), (0, 0))
        else:
            pad_width = ((align, align), (0, 0))
    else:
        if is_3channel:
            pad_width = ((align+1, align), (0, 0), (0, 0))
        else:
            pad_width = ((align+1, align), (0, 0))

    res = np.pad(res, pad_width, 'constant', constant_values=np.median(res.flatten()))
    if h > w:
        res = res.T

    return cv2.resize(res, (export_size, export_size))


def get_qrcode(img_root="./../data_clean/the_real593", lablel_root="./../data_clean/the_real593_label"):
    """

    @param img_root: 圖片資料夾根目錄
    @param lablel_root: 標記資料夾根目錄
    @return: 圈出資料集中每一個qrcode。
    """

    image_paths = glob.glob(img_root+"/*.png")
    lablel_paths = glob.glob(lablel_root+"/*.txt")

    image_paths.sort(key=lambda fnInt: int(Path(fnInt).stem))
    lablel_paths.sort(key=lambda fnInt: int(Path(fnInt).stem))

    def read_txt(fpth):
        # 讀取單個 yolo label file
        res = []

        with open(fpth) as f:
            lines = f.readlines()
            lines = [_.rstrip() for _ in lines]

            for bbox in lines:
                c, x, y, w, h = [_ for _ in bbox.split(' ')]
                res.append((int(c), float(x), float(y), float(w), float(h)))
        return res

    def yolobbox2bbox(x, y, w, h):
        x1, y1 = x - w / 2, y - h / 2
        x2, y2 = x + w / 2, y + h / 2
        return x1, y1, x2, y2

    for img, yolo_txt in zip(image_paths, lablel_paths):
        fn = Path(img).stem
        img = cv2.imread(img)
        w, h = img.shape[1::-1]
        #print(img, yolo_txt)

        qr_yoloLables = read_txt(yolo_txt)

        for idx, yoloL in enumerate(qr_yoloLables):
            if int(yoloL[0]) != 0:
                continue
            x1, y1, x2, y2 = yolobbox2bbox(yoloL[1],yoloL[2],yoloL[3],yoloL[4])
            x1 = int(x1 * w)
            y1 = int(y1 * h)
            x2 = int(x2 * w)
            y2 = int(y2 * h)
            assert os.path.exists("./data/qrCodes")
            try:
                cv2.imwrite(f"./data/qrCodes/{fn}_{idx}.png", img[y1:y2, x1:x2])
            except Exception as e:
                with open("./data/genErrorLog.txt", mode='a', encoding='utf-8') as errF:
                    errF.write(f"Error file:{fn} \t")
                    errF.write(str(e))



if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import cv2

    is_notebook()

    get_qrcode()

    a = cv2.imread("./66.jpg")
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    #a = cv2.imread("./66.jpg", cv2.IMREAD_GRAYSCALE)

    print(a.shape)

    ret = pad_2_square(a, export_size=100)

    print("export shape:", ret.shape)
    plt.imshow(ret)
    plt.show()


