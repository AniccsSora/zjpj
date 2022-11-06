import numpy as np
import cv2

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


if __name__ == "__main__":

    import matplotlib.pyplot as plt
    import cv2

    a = cv2.imread("./66.jpg")
    a = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

    #a = cv2.imread("./66.jpg", cv2.IMREAD_GRAYSCALE)

    print(a.shape)

    ret = pad_2_square(a, export_size=100)

    print("export shape:", ret.shape)
    plt.imshow(ret)
    plt.show()


