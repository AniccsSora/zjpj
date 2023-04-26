import cv2
import numpy as np
from PIL import Image
import qrcode
import os


def np_to_rgb(image, outPIL=False):
    # 檢查圖像通道數
    if len(image.shape) == 2:
        # 果圖像是灰度圖像，則將其轉換為三通道圖像
        image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        # 如果圖像是灰度圖像（但是通道數為 1），則將其轉換為三通道圖像
        image = np.repeat(image, 3, axis=-1)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # 如果圖像是帶有 alpha 通道的四通道圖像，則將其轉換為 RGB 圖像
        image = image[:, :, :3]
    elif len(image.shape) != 3 or image.shape[2] != 3:
        # 如果圖像不是三通道圖像，則拋出異常
        raise ValueError('Unsupported image format')

    # 將 numpy 數組轉換為 PIL Image 對象
    if outPIL:
        image = Image.fromarray(np.uint8(image))
    else:
        image = np.uint8(image)

    return image

def affine_transform(image, same=False):
    """
    簡易的仿射變換
    @param image: 
    @param same: 
    @return: 
    """
    # 檢查圖像類型
    if isinstance(image, str):
        # 如果輸入是檔案路徑，則載入圖像
        assert os.path.exists(image)
        image = cv2.imread(image)
    elif isinstance(image, Image.Image):
        # 如果輸入是 PIL Image 物件，則轉換為 numpy 數組
        image = np.array(image)
    elif isinstance(image, qrcode.image.pil.PilImage):
        # 如果輸入是 qrcode PIL Image 子類別，則轉換為 numpy 數組
        image = np.array(image, dtype=np.uint8) * 255
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        # 如果輸入是 numpy 數組，則直接使用
        pass
    else:
        # 如果輸入類型不正確，則拋出異常
        raise TypeError('Unsupported image type')

    image = np_to_rgb(image)
    # 獲取圖像尺寸
    rows, cols, _ = image.shape

    # 隨機生成旋轉角度和平移距離
    angle = np.random.randint(-180, 180)
    tx = np.random.randint(-50, 50)
    ty = np.random.randint(-50, 50)

    # 計算旋轉後的圖像邊界
    cos = np.abs(np.cos(np.deg2rad(angle)))
    sin = np.abs(np.sin(np.deg2rad(angle)))
    new_cols = int((rows * sin) + (cols * cos))
    new_rows = int((rows * cos) + (cols * sin))
    M_rotation = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
    M_rotation[0, 2] += (new_cols - cols) / 2
    M_rotation[1, 2] += (new_rows - rows) / 2

    # 生成平移矩陣
    M_translation = np.float32([[1, 0, tx], [0, 1, ty]])

    # 進行仿射變換
    image_affine = cv2.warpAffine(image, M_rotation, (new_cols, new_rows))
    image_affine = cv2.warpAffine(image_affine, M_translation, (new_cols, new_rows))

    if same:
        # 裁剪圖像到原始大小
        image_affine = image_affine[
                       int(new_rows / 2 - rows / 2):int(new_rows / 2 + rows / 2),
                       int(new_cols / 2 - cols / 2):int(new_cols / 2 + cols / 2),
                       ]
    return image_affine


def affine_transform_v2(image, random_pts1=True)->np.ndarray:
    """
    透視變換
    @param image:
    @param random_pts1:
    @return:
    """
    # 檢查圖像類型
    if isinstance(image, str):
        assert os.path.exists(image)
        # 如果輸入是檔案路徑，則載入圖像
        image = cv2.imread(image, flags=0)
    elif isinstance(image, Image.Image):
        # 如果輸入是 PIL Image 物件，則轉換為 numpy 數組
        image = np.array(image)
    elif isinstance(image, np.ndarray):
        # 如果輸入是 numpy 數組，則直接使用
        pass
    else:
        # 如果輸入類型不正確，則拋出異常
        raise TypeError('Unsupported image type', type(image))

    # 獲取圖像尺寸
    cols, rows = image.shape[1::-1]

    # 隨機生成四個控制點
    if random_pts1:
        pts1 = np.float32([[np.random.randint(cols // 2), np.random.randint(rows // 2)],
                           [np.random.randint(cols // 2, cols), np.random.randint(rows // 2)],
                           [np.random.randint(cols // 2, cols), np.random.randint(rows // 2, rows)],
                           [np.random.randint(cols // 2), np.random.randint(rows // 2, rows)]])
    else:
        pts1 = np.float32([[0, 0],
                           [cols, 0],
                           [cols, rows],
                           [0, rows]])
    # 隨機生成仿射變換後的目標控制點
    offset_x = np.random.randint(cols // 1)
    offset_y = np.random.randint(rows // 1)

    pts2 = np.float32([[pts1[0][0] + np.random.randint(-offset_x, offset_x + 1),
                        pts1[0][1] + np.random.randint(-offset_y, offset_y + 1)],
                       [pts1[1][0] + np.random.randint(-offset_x, offset_x + 1),
                        pts1[1][1] + np.random.randint(-offset_y, offset_y + 1)],
                       [pts1[2][0] + np.random.randint(-offset_x, offset_x + 1),
                        pts1[2][1] + np.random.randint(-offset_y, offset_y + 1)],
                       [pts1[3][0] + np.random.randint(-offset_x, offset_x + 1),
                        pts1[3][1] + np.random.randint(-offset_y, offset_y + 1)]])
    if False:
        perm = np.random.permutation(pts2.shape[0])
        # 對矩陣的行進行重新排列
        pts2 = pts2[perm, :]

    # print(pts2)
    # print(np.abs(pts2, pts1) / (rows, cols))

    # 計算透視變換矩陣
    M_perspective = cv2.getPerspectiveTransform(pts1, pts2)

    # 進行透視變換
    max_x, max_y = np.max(pts2, axis=0)
    max_x, max_y = int(max_x), int(max_y)
    image_affine = cv2.warpPerspective(image, M_perspective, (max_x, max_y))

    return image_affine

if __name__ == "__main__":
    # v2 會使用透視變換
    # example
    #im = Image.fromarray(affine_transform("./data/camera.png"))
    im = Image.fromarray(affine_transform_v2("./data/camera.png"))
    affine_transform_v2(cv2.imread("./data/camera.png"))
    affine_transform_v2(Image.open("./data/camera.png"))
    affine_transform_v2(Image.open("./data/camera_rgb.png"))

    im.show()



