import model
import torch
import torch.nn as nn
from dataloader.PatchesDataset import get_Dataloader
from torch.nn.functional import one_hot
import numpy as np
import cv2
import argparse
from little_function import cutting_cube, cutting_cube_include_surplus
import matplotlib.pyplot as plt
import matplotlib as mpl

parser = argparse.ArgumentParser("main2 參數設定")
# parser.add_argument("--")
param = parser.parse_args()

def scalling_img(img: np.ndarray, scale_list=[0.3, 0.5, 0.7]):
    w, h = img.shape[1::-1]
    res_dict = dict.fromkeys(scale_list)
    for scale in scale_list:
        re_w, re_h = int(w*scale), int(h*scale)
        img_gauBr = cv2.GaussianBlur(img, (21, 21), 5)
        img_bil = cv2.bilateralFilter(img, 9, 75, 75)
        res_dict[scale] = cv2.resize(img_gauBr, dsize=(re_w, re_h), interpolation=cv2.INTER_AREA)
    return res_dict

def get_tensor_batch(dict_muiltiple_images: dict, dict_xyxy: dict, device):
    """

    """
    assert dict_muiltiple_images.keys() == dict_xyxy.keys()
    dkeys = dict_muiltiple_images.keys()
    res_tensor_dict = dict.fromkeys(dkeys)

    for scale_val in res_tensor_dict.keys():
        tmp_ = []
        image = dict_muiltiple_images[scale_val]
        xyxy_generator = dict_xyxy[scale_val]
        # res_tensor_dict[scale_val] = 這邊塞一個 batch
        for xyxy in xyxy_generator:
            p = image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]  # patch
            tmp_.append(p)
        # 前處理，因應效能問題
        tmp_ = np.array(tmp_)
        # 將一張 scalling image patches 傳成 tensor
        res_tensor_dict[scale_val] = torch.tensor(tmp_, dtype=torch.float32, device=device).view(-1,1,32,32)

    return res_tensor_dict


def get_xyxy_generator_dict(scale_list: list, multiple_scalling_imgs: dict, cube_size, overlap):
    res_dict = dict.fromkeys(scale_list)
    for scale_val, scale_img in multiple_scalling_imgs.items():
        w, h = scale_img.shape[1::-1]
        res_dict[scale_val] = cutting_cube_include_surplus((w, h), cube_size, overlap)
    return res_dict


if __name__ == "__main__":
    # 針對實驗寫迴圈 生出圖片
    pred_img = cv2.imread("./data/paper_qr/File 088.bmp", cv2.IMREAD_GRAYSCALE)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_path = "./log_save/20211210_0021_00_第二次實驗/weight.pt"
    qr_dir = "./data/val_patch_True"
    bg_dir = "./data/val_patch_False"
    val_dataloader = get_Dataloader(qrcode_dir=qr_dir, background_dir=bg_dir)
    net = model.QRCode_CNN()
    net.load_state_dict(torch.load(weight_path))
    net.cuda()
    class_label = ['background', 'QRCode']
    scale_list = [0.3, 0.5, 0.7, 1.0]  # 這被大量參用
    use_0_1 = False  # param
    # 存放著 ndarray
    multiple_scalling_imgs = scalling_img(pred_img, scale_list=scale_list)

    multiple_patch_xyxy = get_xyxy_generator_dict(scale_list, multiple_scalling_imgs,
                                             cube_size=32,
                                             overlap=1.0)

    # for scale_val, scale_img in multiple_scalling_imgs.items():
    #     w, h = scale_img.shape[1::-1]
    #     multiple_patch_xyxy[scale_val] = cutting_cube_include_surplus((w, h), 32, overlap=1.0)

    # predict 紀錄
    predict_dict = dict.fromkeys(scale_list)

    multi_scaling_batch = get_tensor_batch(dict_muiltiple_images=multiple_scalling_imgs,
                                           dict_xyxy=multiple_patch_xyxy,
                                           device=device)

    pred_label_dict = dict.fromkeys(scale_list)  # 各 scalling 預測 class 結果
    pred_label_p_dict = dict.fromkeys(scale_list)  # 各 scalling 預測 為該 class 的機率。
    for scale_val, batch in multi_scaling_batch.items():
        # !!! 這邊是 0~255 下去判斷。 !!!
        if use_0_1:
            pred_p = net(batch / 255.0).softmax(axis=1)
        else:
            pred_p = net(batch).softmax(axis=1)
        pred_p = pred_p.detach().cpu().numpy()
        pred_label = np.argmax(pred_p, axis=1)  # label number
        pred_label_dict[scale_val] = pred_label
        pred_label_p = pred_p.max(axis=1)  # 機率
        pred_label_p_dict[scale_val] = pred_label_p

    # 各scalling 有選到的座標
    pick_xyxy_dict = dict.fromkeys(scale_list, [])
    # 因為用生成器所以必須這樣再一次init
    multiple_patch_xyxy_2st = get_xyxy_generator_dict(scale_list, multiple_scalling_imgs,
                                             cube_size=32,
                                             overlap=1.0)
    # 有了 pred_label 了 接下來將 1 座標篩出來
    for scale_val, pred_c in pred_label_dict.items():
        pick_idx = np.argwhere(pred_c == 1).squeeze()
        all_pred_pr = pred_label_p_dict[scale_val]  # 根據預測的label的期望機率
        #
        for idx, xyxy in enumerate(multiple_patch_xyxy_2st[scale_val]):
            if idx in pick_idx:
                print("預測機率+入:", all_pred_pr[idx])
                pick_xyxy_dict[scale_val].append(xyxy)

    # 將預測的 bbox 繪製到多尺度的圖片上
    with_bbox_multi_images = []  # 存放被畫上 bbox 的 multiple scaling images
    # 各 scalling 的座標點
    for scale_val, xyxy_s in pick_xyxy_dict.items():
        image = multiple_scalling_imgs[scale_val]
        image_ = np.array(image)
        for xyxy in xyxy_s:  # xyxy_s = 多個座標組
            cv2.rectangle(image_, xyxy[0:2], xyxy[2:], color=(255, 0, 0), thickness=1)
        #cv2.imshow(str(scale_val), image_)
        with_bbox_multi_images.append(image_)
    #cv2.waitKey(0)

    # === plt show param setting
    mpl.rcParams["figure.dpi"] = 120  # default: 100
    nrows, ncols = 2, 2  # array of sub-plots
    figsize = [10, 6]  # figure size, inches
    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.canvas.manager.set_window_title("多尺度預測結果")
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        img = with_bbox_multi_images[i]
        axi.imshow(img, alpha=1.0, cmap=plt.get_cmap('gray'))
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # 將 行/列索引 寫為軸的標題以供識別
        # axi.set_title("Row:" + str(rowid) + ", Col:" + str(colid))
        axi.set_title(f"Scale: {str(scale_list[i])}")
    plt.tight_layout()
    plt.show()

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
