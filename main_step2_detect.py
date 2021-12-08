import model
import torch
import torch.nn as nn
from dataloader.PatchesDataset import get_Dataloader
from torch.nn.functional import one_hot
import numpy as np
import cv2
import argparse
from little_function import cutting_cube, cutting_cube_include_surplus

parser = argparse.ArgumentParser("main2 參數設定")
# parser.add_argument("--")
param = parser.parse_args()

def scalling_img(img: np.ndarray, scale_list=[0.3, 0.5, 0.7]):
    res_list = []
    w, h = img.shape[1::-1]
    for scale in scale_list:
        re_w, re_h = int(w*scale), int(h*scale)
        res_list.append(cv2.resize(img, dsize=(re_w, re_h), interpolation=cv2.INTER_AREA))
    return res_list


if __name__ == "__main__":
    pred_img = cv2.imread("./data/paper_qr/File 088.bmp", cv2.IMREAD_GRAYSCALE)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_path = "./log_save/20211206_2247_31/weight.pt"
    qr_dir = "./data/val_patch_True"
    bg_dir = "./data/val_patch_False"
    val_dataloader = get_Dataloader(qrcode_dir=qr_dir, background_dir=bg_dir)
    net = model.QRCode_CNN()
    net.load_state_dict(torch.load(weight_path))
    net.cuda()

    resize_imgs = scalling_img(pred_img)

    for image in resize_imgs:
        w, h = image.shape[1::-1]
        # cube_gen = cutting_cube((w, h), 32, overlap=1.0)
        cube_gen = cutting_cube_include_surplus((w, h), 32, overlap=1.0)
        for xyxy in cube_gen:
            _ = np.array(image)
            drawed = cv2.rectangle(_, xyxy[0:2], xyxy[2:], color=(255,0,0), thickness=1)
            cv2.destroyAllWindows()
            cv2.imshow("show", drawed)
            cv2.waitKey(0)
            #cv2.rectangle(影像, 頂點座標, 對向頂點座標, 顏色, 線條寬度)



    # =============== 計算 loss
    # with torch.no_grad():
    #     loss_criterion = nn.CrossEntropyLoss()
    #     total_loss, cnt = 0, 0
    #     for data in val_dataloader:
    #         images, labels = data
    #         # 因為是練好的 model 預期出來需要是機率
    #         # 並且要跟 one-hot 算 loss，(分布與分布之間的 loss)
    #         pred_y = net(images).softmax(dim=1)
    #         labels = one_hot(labels.long(), num_classes=2)
    #         total_loss += loss_criterion(labels.float(), pred_y)
    #         cnt += 1
    #     avg_loss = total_loss/cnt
    #     print("val_loss:", avg_loss.detach().cpu().numpy())
