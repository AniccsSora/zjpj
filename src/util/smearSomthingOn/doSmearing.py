import random
from PIL import Image, ImageDraw
import random
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import random
import cv2
import io


from PIL import Image, ImageDraw
import numpy as np
from scipy.special import binom
import numpy as np
from numpy.polynomial import polynomial
from scipy.special import comb

import random
import numpy as np
from scipy.special import comb
from PIL import Image, ImageDraw

def draw_bezier_curve(image, points, num_segments=50)->Image.Image:
    """
    读取指定路径的图片并在上面绘制贝塞尔曲线

    :param image: 图片路径
    :param points: 控制点列表
    :param num_segments: 每条曲线的分段数
    :param width: 线宽
    :return: 绘制完曲线的 PIL.Image 对象
    """
    # 读取图片并创建绘图对象

    if isinstance(image, str):
        # 如果輸入是檔案路徑，則載入圖像
        image = Image.open(image)
    elif isinstance(image, Image.Image):
        # 如果輸入是 PIL Image 物件，則直接使用
        if image.mode == 'L':
            image = image.convert('RGB')
        pass
    elif isinstance(image, np.ndarray):
        # 如果輸入是 numpy 數組，轉 PIL Image 物件
        if image.ndim == 2:
            image = np.stack((image,) * 3, axis=-1)
        image = Image.fromarray(image)
    else:
        # 如果輸入類型不正確，則拋出異常
        raise TypeError('Unsupported image type')

    draw = ImageDraw.Draw(image)

    # 绘制贝塞尔曲线
    for i in range(len(points)):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        curve, _ = generate_bezier_curve(points[i], num_segments=num_segments, color=color)
        # random line width
        width = random.randint(int(min(image.size) * 0.02), int(max(image.size) * 0.05))
        draw.line(list(map(tuple, curve)), fill=color, width=width)

    return image

def generate_bezier_curve(points, num_segments=50, color=(255, 0, 0)):
    t = np.linspace(0, 1, num_segments)
    n = len(points) - 1
    bernstein = np.array([comb(n, i) * t**i * (1 - t)**(n-i) for i in range(n + 1)])
    x = np.dot(points[:, 0], bernstein)
    y = np.dot(points[:, 1], bernstein)
    return np.stack((x, y), axis=-1), color


def generate_random_points(w, h, n):
    """
    生成随机坐标

    :param w: 坐标范围宽度
    :param h: 坐标范围高度
    :param n: 坐标数量
    :return: 随机坐标的 numpy 数组
    """
    x = np.random.randint(w, size=n)
    y = np.random.randint(h, size=n)
    points = np.stack((x, y), axis=-1)
    return points

def generate_random_points_near_edges(image, n):
    """
    生成位于图像四个角落和边缘附近的随机点

    :param image: PIL.Image 对象，表示输入图像
    :param n: 生成的随机点数量
    :param offset: 随机点距离图像边缘和角落的最小距离
    :return: 随机点的 numpy 数组
    """
    # 获取图像宽度和高度
    w, h = image.size

    # 生成位于图像四个角落的固定点
    corner_points = np.array([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])

    offset = lambda: random.randint(1, 10)

    # 生成位于图像四个边缘的随机点
    edge_points_top = np.stack((np.random.randint(offset(), w-offset(), size=n), np.zeros(n)), axis=-1)
    edge_points_bottom = np.stack((np.random.randint(offset(), w-offset(), size=n), np.full(n, h-1)), axis=-1)
    edge_points_left = np.stack((np.zeros(n), np.random.randint(offset(), h-offset(), size=n)), axis=-1)
    edge_points_right = np.stack((np.full(n, w-1), np.random.randint(offset(), h-offset(), size=n)), axis=-1)
    edge_points = np.concatenate((edge_points_top, edge_points_bottom, edge_points_left, edge_points_right))

    # 生成位于图像中心区域的随机点
    #center_points = np.stack((np.random.randint(offset, w-offset, size=n*2), np.random.randint(offset, h-offset, size=n*2)), axis=-1)

    # 将所有点组合成一个数组
    #points = np.concatenate((corner_points, edge_points, center_points))
    points = np.concatenate((corner_points, edge_points))
    points = points.astype(int)
    #
    perm = np.random.permutation(points.shape[0])
    #points[np.random.permutation(points.shape[0]), :]
    matrix_permuted = points[perm, :]

    return matrix_permuted[:n]

def gen_curves_points_list(ref_image, n, complixty=(3,11)):
    """

    @param ref_image: 參考圖片
    @param n: 生成的組數
    @param complixty: 線條複雜度範圍
    @return:
    """
    res = []
    for _ in range(n):
        __aaa = generate_random_points_near_edges(ref_image, random.randint(complixty[0], complixty[1]))
        res.append(__aaa)
    return res

if __name__ == "__main__":
    # 加载示例图像
    image = Image.open('./data/0_0.png')
    #image = './data/single_qr/0_0.png'
    #image = cv2.imread('./data/single_qr/0_0.png', 0)
    #image = Image.fromarray(image)

    # Generate some random control points for two curves
    #points1 = np.array([(50, 50), (100, 150), (150, 50), (200, 100)])
    #points2 = np.array([(50, 150), (100, 50), (150, 150), (200, 50), (22, 22)])

    # 在圖片範圍內生成幾個點
    # points1 = generate_random_points_near_edges(image, 7)
    # points2 = generate_random_points_near_edges(image, 9)
    # points3 = generate_random_points_near_edges(image, 11)
    #
    # have_cur = draw_bezier_curve(image, [points1, points2, points3])

    #
    points = gen_curves_points_list(
        ref_image=image,
        n=3,
        complixty=(3, 19)
    )

    have_cur = draw_bezier_curve(image, points)

    have_cur.save("./output.png")



