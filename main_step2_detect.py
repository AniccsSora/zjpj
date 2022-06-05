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
import os
from merge_bbx import merge_bboxes
os.environ['KMP_DUPLICATE_LIB_OK']='True'  # 暫且安捏
parser = argparse.ArgumentParser("main2 參數設定")
# parser.add_argument("--")
param = parser.parse_args()

def scalling_img(img: np.ndarray, scale_list=[0.3, 0.5, 0.7]):
    w, h = img.shape[1::-1]
    res_dict = dict.fromkeys(scale_list)
    for scale in scale_list:
        re_w, re_h = int(w*scale), int(h*scale)
        img_gauBr = cv2.GaussianBlur(img, (5, 5), 3)
        # img_bil = cv2.bilateralFilter(img, 9, 75, 75)
        res_dict[scale] = cv2.resize(img_gauBr, dsize=(re_w, re_h), interpolation=cv2.INTER_AREA)
        #res_dict[scale] = cv2.resize(img, dsize=(re_w, re_h), interpolation=cv2.INTER_AREA)
    return res_dict

def get_tensor_batch(dict_muiltiple_images: dict, dict_xyxy: dict, device):
    """
    回傳各 scale 的 patch，與其對應座標
    """
    assert dict_muiltiple_images.keys() == dict_xyxy.keys()
    dkeys = dict_muiltiple_images.keys()
    res_tensor_dict = dict.fromkeys(dkeys)
    res_xyxy_dict = dict.fromkeys(dkeys)

    for scale_val in res_tensor_dict.keys():
        tmp_ = []
        tmp_xyxy_ = []
        image = dict_muiltiple_images[scale_val]
        xyxy_generator = dict_xyxy[scale_val]
        # res_tensor_dict[scale_val] = 這邊塞一個 batch
        for xyxy in xyxy_generator:
            p = image[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]  # patch
            tmp_xyxy_.append(xyxy)
            tmp_.append(p)
        # 前處理，因應效能問題
        tmp_ = np.array(tmp_)
        tmp_xyxy_ = list(tmp_xyxy_)
        # 將一張 scalling image patches 傳成 tensor
        res_tensor_dict[scale_val] = torch.tensor(tmp_, dtype=torch.float32, device=device).view(-1,1,32,32)
        res_xyxy_dict[scale_val] = tmp_xyxy_
    # end for-loop
    return res_tensor_dict, res_xyxy_dict


def get_xyxy_generator_dict(scale_list: list, multiple_scalling_imgs: dict, cube_size, overlap):
    res_dict = dict.fromkeys(scale_list)
    for scale_val, scale_img in multiple_scalling_imgs.items():
        w, h = scale_img.shape[1::-1]
        res_dict[scale_val] = cutting_cube_include_surplus((w, h), cube_size, overlap)
    return res_dict

def determine_new_p(p_list, percentile, bigger_1_reduce_percentile=1):
    """
    @param p_list: 預測框的所有機率
    @param percentile: 要取的百分位數
    @return: 估計最佳閥值
    """
    res = None
    p_list = p_list[p_list > 0.5]  # 閥值大於 0.5 才會被取用
    if len(p_list[p_list <= 0.5]) == 0:
        print(f'[debug]: 過濾閥值均大於 0.5')
    else:
        print(f'[debug]: 過濾閥值 "沒有" 均大於 0.5')
    res = np.percentile(p_list, percentile)
    while res >= 1.0:
        percentile -= 1
        res = np.percentile(p_list, percentile)
    print(f'更新閥值為: {res}, 使用 {percentile} 百分位值')
    return res, percentile


if __name__ == "__main__":
    bbox_threshold = 0.995  # 0.99999995
    auto_thres = True  # False=使用固定閥值
    percentile_pick = 99.999  # 動態閥值的取的百分位數

    thick = 2
    overlap = 0.3  # 切割精細度
    # merge 策略 (float)
    merge_delta_x = 0.01
    merge_delta_y = 0.01
    # 針對實驗寫迴圈 生出圖片
    #pred_img = cv2.imread("./data/paper_qr/File 088.bmp", cv2.IMREAD_GRAYSCALE)
    #pred_img = cv2.imread("./data/raw_qr/qr_0016_big.jpg", cv2.IMREAD_GRAYSCALE)
    #pred_img = cv2.imread("./data/raw_qr/qr_0013.jpg", cv2.IMREAD_GRAYSCALE)
    pred_img = cv2.imread("./data_clean/the_real593/41.png", cv2.IMREAD_GRAYSCALE)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    weight_path = "./log_save/20220425_0233_02_batch_256/weight.pt"
    qr_dir = "./data/val_patch_True"
    bg_dir = "./data/val_patch_False"
    val_dataloader = get_Dataloader(qrcode_dir=qr_dir, background_dir=bg_dir)
    net = model.QRCode_CNN()
    net.load_state_dict(torch.load(weight_path))
    net.cuda()
    class_label = ['background', 'QRCode']
    #scale_list = [0.2, 0.3, 0.5, 0.75]  # 這被大量參用
    scale_list = [0.3, 0.4, 0.5, 0.6]  # 這被大量參用
    use_0_1 = True  # param
    # 存放著 不同 scale 的 src image.
    multiple_scalling_imgs = scalling_img(pred_img, scale_list=scale_list)
    # 不同 scale 的 32x32 xyxy。
    multiple_patch_xyxy = get_xyxy_generator_dict(scale_list, multiple_scalling_imgs,
                                             cube_size=32,
                                             overlap=overlap)

    # predict 紀錄
    predict_dict = dict.fromkeys(scale_list)

    multi_scaling_patches, multi_xyxy_dict = get_tensor_batch(dict_muiltiple_images=multiple_scalling_imgs,
                                             dict_xyxy=multiple_patch_xyxy,
                                             device=device)

    # 預測類別
    pred_label_dict = dict.fromkeys(scale_list)  # 各 scalling 預測 class 結果
    # 預測該類別的機率
    pred_label_p_dict = dict.fromkeys(scale_list)  # 各 scalling 預測 為該 class 的機率。
    # 他的座標
    pred_patch_xyxy_dict = multi_xyxy_dict

    # 預測各 scale 的 pathces 機率。
    # 這邊只有 label 機率，以及該 patch image.
    for scale_val, batch in multi_scaling_patches.items():
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
    pick_xyxy_dict = dict.fromkeys(scale_list, None)

    # debug，紀錄各尺度篩出來的 Qrcode數量
    qrcode_bbox_cnt_dict = dict().fromkeys(scale_list, 0)
    # debug，紀錄為 qrcode 預選框的數量，要再進一步篩選過低期望的數值。
    preselection_number_dict = dict().fromkeys(scale_list, 0)
    # debug，被剔除預選框的數量
    non_select_number_dict = dict().fromkeys(scale_list, 0)
    # debug，紀錄最後採用的百分位值 (百分位)
    percent_pick_dict = dict().fromkeys(scale_list, 0)
    # debug，紀錄最後採用的百分位值閥值 (使用的預測機率)
    percent_pick_predictP_dict = dict().fromkeys(scale_list, 0)

    # 有了 pred_label 了 接下來將屬於 (1 class, QRCode)的座標篩出來
    for scale_val, pred_c in pred_label_dict.items():
        # scale_val尺度下所預測為該 label 期望值
        all_pred_pr = pred_label_p_dict[scale_val]
        # 取得 label == 1 的 args
        qr_label_indices = np.squeeze(np.argwhere(pred_label_dict[scale_val] == 1))
        # 預選框數量
        preselection_number_dict[scale_val] = len(qr_label_indices)
        # 拿出他的 xyxy
        _xyxy_tmp_ = []
        # 決定是否使用自動閥值
        if auto_thres:
            bbox_threshold, percent_pick = determine_new_p(all_pred_pr, percentile_pick)
            print(f"{scale_val} use NEW THRESHOLD {bbox_threshold}, "
                  f"determine from {percent_pick} ppercentile.")
            print("===============================================")
            percent_pick_dict[scale_val] = percent_pick  # 使用的百分位數
            percent_pick_predictP_dict[scale_val] = bbox_threshold  # 百分位數對應的閥值
        # 自動篩選 threshold 完畢

        for is_qr_idx in qr_label_indices:
            _p = all_pred_pr[is_qr_idx]  # 期望值
            if _p > bbox_threshold:
                # pred_patch_xyxy_dict[scale_val] : scale_val尺度下，被切出的所有 32x32 座標
                xyxy = pred_patch_xyxy_dict[scale_val][is_qr_idx]
                _xyxy_tmp_.append(xyxy)
                #print("預測機率加入:", _p)
                qrcode_bbox_cnt_dict[scale_val] += 1
            else:
                #print("不加入:", _p)
                non_select_number_dict[scale_val] += 1
        # 上方 for-loop 篩選不符合資格的 bbox 完畢

        # 篩掉期望值不夠的 bbox
        pick_xyxy_dict[scale_val] = _xyxy_tmp_
    print("===============================================")

    # debug :
    for scale in scale_list:
        print("{} 預測出 {:3.0f}/{:3.0f} QRCode patches. (預選框:{:2.0f}) (剔除框:{:2.0f})".
              format(scale,
                     qrcode_bbox_cnt_dict[scale],
                     len(pred_label_p_dict[scale]),
                     preselection_number_dict[scale],
                     non_select_number_dict[scale])
              )
    # 將預測的 bbox 繪製到多尺度的圖片上
    with_bbox_multi_images = []  # 存放被畫上 bbox 的 multiple scaling images
    with_merged_bbox_multi_images = []  # 存放被 merge 過的 bbox 的 multiple scaling images
    for scale_val in scale_list:
        image = multiple_scalling_imgs[scale_val]
        image_ = np.array(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
        image_m_ = np.array(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))#np.array(image)
        # 一五一十地畫出來
        for xyxy in pick_xyxy_dict[scale_val]:
            cv2.rectangle(image_, xyxy[0:2], xyxy[2:], color=(255, 0, 0), thickness=thick)
        # 要繪製之前 先 merge
        for xyxy in merge_bboxes(pick_xyxy_dict[scale_val],
                                 delta_x=merge_delta_x,
                                 delta_y=merge_delta_y):
            cv2.rectangle(image_m_, xyxy[0:2], xyxy[2:], color=(0, 255, 0), thickness=thick)
        with_bbox_multi_images.append(image_)
        with_merged_bbox_multi_images.append(image_m_)

    # === plt show param setting
    mpl.rcParams["figure.dpi"] = 120  # default: 100
    nrows, ncols = 2, 2  # array of sub-plots
    figsize = [10, 6]  # figure size, inches
    # create figure (fig), and array of axes (ax)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.canvas.manager.set_window_title("多尺度預測結果")
    fig.suptitle('QR-Patch prediction result', fontsize=12)
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
        if auto_thres:
            # 使用動態thres 需要印出更詳細數值
            cur_scale = scale_list[i]
            _percentile = str(percent_pick_dict[cur_scale])  # 使用的百分位數
            _thres_P = str(percent_pick_predictP_dict[cur_scale])  # 百分位數對應的閥值
            axi.set_title(f"Scale: {str(cur_scale)}, percentile:({_percentile})={_thres_P}")
        else:
            axi.set_title(f"Scale: {str(scale_list[i])}")
    plt.tight_layout()
    plt.savefig('detection.png')
    # plt.show()


    # 繪製 merge bbox 版本
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    fig.canvas.manager.set_window_title("多尺度預測結果_結合 bbox 版")
    fig.suptitle('Merge BBox result', fontsize=12)
    for i, axi in enumerate(ax.flat):
        # i runs from 0 to (nrows*ncols-1)
        # axi is equivalent with ax[rowid][colid]
        img = with_merged_bbox_multi_images[i]
        axi.imshow(img, alpha=1.0, cmap=plt.get_cmap('gray'))
        # get indices of row/column
        rowid = i // ncols
        colid = i % ncols
        # 將 行/列索引 寫為軸的標題以供識別
        # axi.set_title("Row:" + str(rowid) + ", Col:" + str(colid))
        if auto_thres:
            # 使用動態thres 需要印出更詳細數值
            cur_scale = scale_list[i]
            _percentile = str(percent_pick_dict[cur_scale])  # 使用的百分位數
            _thres_P = str(percent_pick_predictP_dict[cur_scale])  # 百分位數對應的閥值
            axi.set_title(f"Scale: {str(cur_scale)}, percentile:({_percentile})={_thres_P}")
        else:
            axi.set_title(f"Scale: {str(scale_list[i])}")
    plt.tight_layout()
    plt.savefig('detection_merge_ver.png')
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
