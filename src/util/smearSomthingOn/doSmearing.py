import random
from PIL import Image, ImageDraw
import cv2


# 定义绘图函数
def draw_random_line(image, color, width):
    # 随机生成线条起点和终点
    x1 = random.randint(0, image.width)
    y1 = random.randint(0, image.height)
    x2 = random.randint(0, image.width)
    y2 = random.randint(0, image.height)

    # 在起点和终点之间绘制手绘线条
    draw = ImageDraw.Draw(image)
    draw.line([(x1, y1), (x2, y2)], fill=color, width=width)


if __name__ == "__main__":
    # 加载图像文件
    # 加载示例图像
    img = cv2.imread(cv2.samples.findFile('./data/single_qr/0_0.png'))
    image = Image.fromarray(img)

    # 在图像上随机绘制多条手绘线条
    for i in range(10):
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        width = random.randint(1, 5)
        draw_random_line(image, color, width)

    # 保存绘制后的图像
    image.save("example_with_lines.png")