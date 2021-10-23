from cv2 import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_patch_switch_frequency(src, condi=4):
    """
    檢查 qr code patch 的 frequency 大小，評估 qr code 大約是幾個 module(佔據幾個 pixel) 組成。
    @ param: condi, 每邊選幾個 line 作為計算標準。因為有 2邊所以會選擇出 condi*2 個評估 line。
    """
    assert src.shape[0] == src.shape[1]  # 要求為方形圖片
    assert src.ndim == 2  # 只支援單通道圖片
    assert condi >= 3  # 就是要大於等於 3 ~
    assert condi <= 32  # 基於論文他就是 32x32 的patch~

    trip = src.shape[0] // (condi-1)
    # 取 col, row
    cols_lines, row_lines = src[:, 0::trip], src[0::trip, :]

    row_lines = row_lines.T

    lines = np.hstack((cols_lines, row_lines))

    #  二值化 (1亮，0暗)
    lines = np.array(lines > 127, np.uint8)
    lines = lines.T

    # 計算用 子子函式
    def cnt_1d_vec_freq(vec):
        """
        @param vec: 1d itter
        @return: list
        """
        resl = []
        cur = vec[0]  # 取該 line 第一個
        cnt = 0
        for e in vec:
            if e == cur:
                cnt += 1
            else:
                cur = e
                resl.append(cnt)
                cnt = 1
        else:
            resl.append(cnt)

        return resl
    # 總結評估方法
    res = []
    for line in lines:
        res.append(cnt_1d_vec_freq(line))

    eva_median = 0
    eva_mean = 0
    eva_number = 0
    # res長度為 3 以上(包括) ，取中位數
    for r in res:
        if len(r) >= 3:
            eva_median += np.median(r)
            eva_mean += np.mean(r)
            eva_number += 1
    eva_median /= eva_number
    eva_mean /= eva_number

    # 平均後無條件進位
    return int(np.ceil((eva_median + eva_mean) / 2))


def get_square_region(src, xy: tuple, size):
    """
    @param src: ndarray
    @param xy: xy
    @return: sub image
    """
    size = 32
    w, h = src.shape[1::-1]
    x, y = xy[0], xy[1]
    y = np.clip(y, 0, h - 1 - size)
    x = np.clip(x, 0, w - 1 - size)
    return src[y:size+y, x:size+x]


if __name__ == "__main__":


    # 這張圖平率是 10
    img = cv2.imread('./data/bin-qrcode.png', cv2.COLOR_BGR2GRAY)

    size = 32
    w, h = img.shape[1::-1]
    x, y = np.clip(500, 0, w-1-size), np.clip(300, 0, h-1-size)

    sub = get_square_region(img, (x, y), size)

    # calculate_patch_switch_frequency 使用方法
    module_size = calculate_patch_switch_frequency(sub, condi=32)

    print(module_size)


