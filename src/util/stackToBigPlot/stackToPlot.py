import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

def create_image_grid(rows, cols, image_list):
    assert rows*cols == len(image_list)

    # Calculate the dimensions of each image in the grid
    max_size = 0
    for _ in image_list:
        height, width, _ = cv2.imread(str(_)).shape
        max_size = max(height, width)

    # Create an empty grid image to store the stacked images
    grid_image = np.zeros((rows * max_size, cols * max_size, 3), dtype=np.uint8)

    # Loop over the image list and add each image to the grid
    for i, image_path in enumerate(image_list):
        # Load the image and resize it to the maximum size
        image = cv2.imread(str(image_path))
        h, w, _ = image.shape
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
        image = cv2.resize(image, (new_w, new_h))

        # Calculate the row and column position of the image in the grid
        row = i // cols
        col = i % cols

        # Calculate the start and end positions of the image in the grid
        start_h = row * max_size
        end_h = start_h + new_h
        start_w = col * max_size
        end_w = start_w + new_w

        # Add the image to the grid
        grid_image[start_h:end_h, start_w:end_w, :] = image
    grid_image = cv2.cvtColor(grid_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(grid_image)

def create_image_grid_by_MATPLT(rows, cols, image_list):
    assert rows*cols == len(image_list)

    # Calculate the dimensions of each image in the grid
    max_size = 0
    for _ in image_list:
        height, width, _ = plt.imread(str(_)).shape
        max_size = max(max_size, max(height, width))

    # Create a figure and a grid of subplots
    fig, axs = plt.subplots(rows, cols, figsize=(cols, rows))

    # Loop over the image list and add each image to the grid
    for i, image_path in enumerate(image_list):
        # Load the image and resize it to the maximum size
        image = plt.imread(str(image_path))
        h, w, _ = image.shape
        if h > w:
            new_h = max_size
            new_w = int(w * (max_size / h))
        else:
            new_w = max_size
            new_h = int(h * (max_size / w))
        image = cv2.resize(image, (new_w, new_h))

        # Calculate the row and column position of the image in the grid
        row = i // cols
        col = i % cols

        # Add the image to the appropriate subplot
        axs[row, col].imshow(image)
        axs[row, col].set_xticks([])
        axs[row, col].set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)


if __name__ == "__main__":
    images_dir = Path("../makeGoodAffineQR/data/affine")
    assert images_dir.is_dir()

    images_path = [_ for _ in images_dir.rglob("*.*")]

    rows, cols = 10, 10

    # 緊湊繪製
    big_stack_img = create_image_grid(rows, cols, images_path)
    big_stack_img.show()

    # 清晰圖表
    big_stack_img = create_image_grid_by_MATPLT(rows, cols, images_path)
    plt.show()

