import logging
import os.path
import json
import copy
import model as qrcnn_model
import torch
from misc.F import set_logger
import glob
import numpy as np
import cv2
from little_function import cutting_cube_include_surplus
from merge_bbx import merge_bboxes
import matplotlib.pyplot as plt
import matplotlib as mpl
from misc.F import ensure_folder, timestamp
from pathlib import Path
from decimal import Decimal
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

LOG_SAVE_FOLDER = f"log_save/{timestamp()}"

ensure_folder(LOG_SAVE_FOLDER)

set_logger(path=LOG_SAVE_FOLDER)
logger = logging.getLogger("Detector_show4d.py")

class Detector:
    def __init__(self, weight, net, save_folder):
        """
        @param weight: .pt 路徑
        @param net: 使用的 nn.Module
        """
        self.save_folder = save_folder
        self.thick = 2  # bbox bound
        self.overlap = 0.3  # 切割精細度
        # merge 策略 (float)
        self.merge_delta_x = 0.01
        self.merge_delta_y = 0.01
        # 圖片金字塔縮放比例
        self.scale_list = [0.25, 0.5, 0.75, 1.0]
        # 使用固定閥值 是與否?
        self.auto_thres = True  # False=使用固定閥值
        # 動態閥值的取的百分位數
        self.percentile_pick = 90
        #
        # 固定定義
        self._class_label = ['background', 'QRCode']
        # 灰階值正規化
        self.grey_normalization = True
        # 切分 patch 的基本大小
        self.cube_size = 32
        #
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        #
        # 辨識時允許的最大邊長，如果超出會縮放到最長邊 < ALLOW_MAXSIZE 為止。
        self.ALLOW_MAXSIZE = 600
        # ====
        #
        self.weight = weight
        self.net = net
        #
        # ===
        self._do_init()

    def _do_init(self):
        self.net.load_state_dict(torch.load(self.weight))
        logger.info(f"Load weight from {self.weight}")
        if torch.cuda.is_available():
            self.net.cuda()
            logger.info(f"Move net to Cuda.")
        else:
            logger.warning("NOT USING CUDA!")

    def scalling_img(self, img: np.ndarray):
        w, h = img.shape[1::-1]
        res_dict = dict.fromkeys(self.scale_list)
        for scale in self.scale_list:
            re_w, re_h = int(w * scale), int(h * scale)
            _cnt = 0
            while re_w <= 32 or re_h <= 32:
                _cnt += 1
                re_w, re_h = int(w * (scale+0.05*_cnt)), int(h * (scale+0.05*_cnt))
            img_gauBr = cv2.GaussianBlur(img, (5, 5), 3)
            # img_bil = cv2.bilateralFilter(img, 9, 75, 75)
            res_dict[scale] = cv2.resize(img_gauBr, dsize=(re_w, re_h), interpolation=cv2.INTER_AREA)
            # res_dict[scale] = cv2.resize(img, dsize=(re_w, re_h), interpolation=cv2.INTER_AREA)
        return res_dict

    def get_xyxy_generator_dict(self, multiple_scalling_imgs: dict):
        res_dict = dict.fromkeys(self.scale_list)
        for scale_val, scale_img in multiple_scalling_imgs.items():
            w, h = scale_img.shape[1::-1]
            res_dict[scale_val] = cutting_cube_include_surplus((w, h), self.cube_size, self.overlap)
        return res_dict

    def get_tensor_batch(self, dict_muiltiple_images: dict, dict_xyxy: dict):
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
            res_tensor_dict[scale_val] = torch.tensor(tmp_, dtype=torch.float32, device=self.device).view(-1, 1, 32, 32)
            res_xyxy_dict[scale_val] = tmp_xyxy_
        # end for-loop
        return res_tensor_dict, res_xyxy_dict

    @staticmethod
    def determine_new_p(p_list, percentile, bigger_1_reduce_percentile=1):
        """
        @param p_list: 預測框的所有機率
        @param percentile: 要取的百分位數
        @return: 估計最佳閥值
        """
        logger = logging.getLogger("[static] Detector.determine_new_p")
        res = None
        p_list = p_list[p_list > 0.5]  # 閥值大於 0.5 才會被取用
        if len(p_list[p_list <= 0.5]) == 0:
            logger.info(f'[debug]: 過濾閥值均大於 0.5')
        else:
            logger.info(f'[debug]: 過濾閥值 "沒有" 均大於 0.5')
        res = np.percentile(p_list, percentile)
        max_limit = 20
        origin_percentile = percentile
        cnt = 0
        while res >= 1.0:
            cnt += 1
            if cnt > max_limit:
                res = np.percentile(p_list, origin_percentile)
                break
            percentile -= 1
            try:
                res = np.percentile(p_list, percentile)
            except ValueError:
                logger.info(f"[warn]: p_list, percentile: {p_list}, {percentile}")
                logger.info("[warn] 自動 clip 到 :", np.clip(percentile, 0, 100))
                res = np.percentile(p_list, np.clip(percentile, 0, 100))
                print("res =", res)
        logger.info(f'更新閥值為: {res}, 使用 {percentile} 百分位值')
        return res, percentile

    def check_resize(self, img: np.ndarray):
        logger = logging.getLogger('Detector.check_resize')
        w, h = img.shape[1::-1]

        if not w < self.ALLOW_MAXSIZE or not h < self.ALLOW_MAXSIZE:
            w_ratio, h_ratio = self.ALLOW_MAXSIZE/w, self.ALLOW_MAXSIZE/h
            w_ratio if w_ratio < h_ratio else h_ratio
            logger.info(f"{self.cursor_img_name} 重新縮放成 w,h:({int(w * w_ratio)}, {int(h * w_ratio)})")
            return cv2.resize(img, (int(w*w_ratio), int(h*w_ratio)))
        return img

    def multiscale_prediction(self, image_path):
        """
        執行輸入圖片路徑的多尺度預測

        @param image_path:
        @return:
        """
        logger = logging.getLogger("Detector.multiscale_prediction")
        assert Path(image_path).is_file()
        pred_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        assert pred_img is not None
        self.cursor_img_name = Path(image_path).stem  # 處理到目前的檔名
        pred_img = self.check_resize(pred_img)
        #
        multiple_scalling_imgs = self.scalling_img(pred_img)
        # self.scalling_img(pred_img)[0.25].shape
        #
        multiple_patch_xyxy = self.get_xyxy_generator_dict(multiple_scalling_imgs)
        #
        multi_scaling_patches, pred_patch_xyxy_dict = self.get_tensor_batch(dict_muiltiple_images=multiple_scalling_imgs,
                                                                  dict_xyxy=multiple_patch_xyxy)
        # 預測類別
        pred_label_dict = dict.fromkeys(self.scale_list)  # 各 scalling 預測 class 結果
        # 預測該類別的機率
        pred_label_p_dict = dict.fromkeys(self.scale_list)  # 各 scalling 預測 為該 class 的機率。

        # 預測各 scale 的 pathces 機率。
        # 這邊只有 label 機率，以及該 patch image.
        for scale_val, batch in multi_scaling_patches.items():
            # !!! 這邊是 0~255 下去判斷。 !!!
            if self.grey_normalization:
                pred_p = self.net(batch / 255.0).softmax(axis=1)
            else:
                pred_p = self.net(batch).softmax(axis=1)
            pred_p = pred_p.detach().cpu().numpy()
            pred_label = np.argmax(pred_p, axis=1)  # label number
            pred_label_dict[scale_val] = pred_label
            pred_label_p = pred_p.max(axis=1)  # 機率
            pred_label_p_dict[scale_val] = pred_label_p

        # 各scalling 有選到的座標
        pick_xyxy_dict = dict.fromkeys(self.scale_list, None)

        # debug，紀錄各尺度篩出來的 Qrcode數量
        qrcode_bbox_cnt_dict = dict().fromkeys(self.scale_list, 0)
        # debug，紀錄為 qrcode 預選框的數量，要再進一步篩選過低期望的數值。
        preselection_number_dict = dict().fromkeys(self.scale_list, 0)
        # debug，被剔除預選框的數量
        non_select_number_dict = dict().fromkeys(self.scale_list, 0)
        # debug，紀錄最後採用的百分位值 (百分位)
        self.percent_pick_dict = dict().fromkeys(self.scale_list, 0)
        # debug，紀錄最後採用的百分位值閥值 (使用的預測機率)
        self.percent_pick_predictP_dict = dict().fromkeys(self.scale_list, 0)

        # 有了 pred_label 了 接下來將屬於 (1 class, QRCode)的座標篩出來
        for scale_val, pred_c in pred_label_dict.items():
            # scale_val尺度下所預測為該 label 期望值
            all_pred_pr = pred_label_p_dict[scale_val]
            # 取得 label == 1 的 args
            qr_label_indices = np.atleast_1d(np.squeeze(np.argwhere(pred_label_dict[scale_val] == 1)))
            # 預選框數量
            preselection_number_dict[scale_val] = len(np.atleast_1d(qr_label_indices))
            # 拿出他的 xyxy
            _xyxy_tmp_ = []
            # 決定是否使用自動閥值
            if self.auto_thres:
                bbox_threshold, percent_pick = self.determine_new_p(all_pred_pr, self.percentile_pick)
                logger.info(f"{scale_val} use NEW THRESHOLD {bbox_threshold}, "
                      f"determine from {percent_pick} ppercentile.")
                print("===============================================")
                self.percent_pick_dict[scale_val] = percent_pick  # 使用的百分位數
                self.percent_pick_predictP_dict[scale_val] = bbox_threshold  # 百分位數對應的閥值
            # 自動篩選 threshold 完畢

            for is_qr_idx in qr_label_indices:
                _p = all_pred_pr[is_qr_idx]  # 期望值
                if _p > bbox_threshold:
                    # pred_patch_xyxy_dict[scale_val] : scale_val尺度下，被切出的所有 32x32 座標
                    xyxy = pred_patch_xyxy_dict[scale_val][is_qr_idx]
                    _xyxy_tmp_.append(xyxy)
                    # print("預測機率加入:", _p)
                    qrcode_bbox_cnt_dict[scale_val] += 1
                else:
                    # print("不加入:", _p)
                    non_select_number_dict[scale_val] += 1
            # 上方 for-loop 篩選不符合資格的 bbox 完畢

            # 篩掉期望值不夠的 bbox
            pick_xyxy_dict[scale_val] = _xyxy_tmp_
        print("===============================================")

        # debug :
        for scale in self.scale_list:
            logger.info("{} 預測出 {:3.0f}/{:3.0f} QRCode patches. (預選框:{:2.0f}) (剔除框:{:2.0f})".
                  format(scale,
                         qrcode_bbox_cnt_dict[scale],
                         len(pred_label_p_dict[scale]),
                         preselection_number_dict[scale],
                         non_select_number_dict[scale])
                  )
        #
        # 將預測的 bbox 繪製到多尺度的圖片上
        self.with_bbox_multi_images = []  # 存放被畫上 bbox 的 multiple scaling images
        self.with_merged_bbox_multi_images = []  # 存放被 merge 過的 bbox 的 multiple scaling images
        self.with_merged_bbox_and_corresponding_img = dict().fromkeys(self.scale_list, None)  # 存放 merged 後的 框框 與對應的圖片

        # 製作後續要優化的 合成bbox 完成之框，並記錄下來
        for scale_val in self.scale_list:
            self.with_merged_bbox_and_corresponding_img[scale_val] = \
                dict({"bbox": [],
                       "img": None
                       })  # 紀錄合成完畢的 bboxes

        for scale_val in self.scale_list:
            image = multiple_scalling_imgs[scale_val]
            self.with_merged_bbox_and_corresponding_img[scale_val]['img']=\
                np.array(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            image_ = np.array(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))
            image_m_ = np.array(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))  # np.array(image)
            # 一五一十地畫出來
            for xyxy in pick_xyxy_dict[scale_val]:
                cv2.rectangle(image_, xyxy[0:2], xyxy[2:], color=(255, 0, 0), thickness=self.thick)
            # 要繪製之前 先 merge
            for xyxy in merge_bboxes(pick_xyxy_dict[scale_val],
                                     delta_x=self.merge_delta_x,
                                     delta_y=self.merge_delta_y):
                self.with_merged_bbox_and_corresponding_img[scale_val]["bbox"].append(xyxy)
                cv2.rectangle(image_m_, xyxy[0:2], xyxy[2:], color=(0, 255, 0), thickness=self.thick)
            self.with_bbox_multi_images.append(image_)
            self.with_merged_bbox_multi_images.append(image_m_)


    def plot_result(self, showit=True):
        """
        將 預測過的結果 繪製出來
        @return:
        """
        logger = logging.getLogger("Detector.plot_result")
        try:
            getattr(self, 'with_bbox_multi_images')
        except AttributeError:
            logger.error("請先呼叫 multiscale_prediction().")
            return
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
            img = self.with_bbox_multi_images[i]
            axi.imshow(img, alpha=1.0, cmap=plt.get_cmap('gray'))
            # get indices of row/column
            rowid = i // ncols
            colid = i % ncols
            # 將 行/列索引 寫為軸的標題以供識別
            # axi.set_title("Row:" + str(rowid) + ", Col:" + str(colid))
            if self.auto_thres:
                # 使用動態thres 需要印出更詳細數值
                cur_scale = self.scale_list[i]
                _percentile = str(self.percent_pick_dict[cur_scale])  # 使用的百分位數
                _thres_P = str(self.percent_pick_predictP_dict[cur_scale])  # 百分位數對應的閥值
                ss_percentile = "{:.0f}".format(Decimal(str(_percentile)))
                ss_thres_P = "{:E}".format(Decimal(str(_thres_P)))
                axi.set_title(f"Scale: {str(cur_scale)}, percentile:({ss_percentile})\n={ss_thres_P}")
            else:
                axi.set_title(f"Scale: {str(self.scale_list[i])}")
        plt.tight_layout()
        plt.savefig(Path(self.save_folder).joinpath(f'{self.cursor_img_name}_detection.png'))
        if showit:
            pass
        else:
            plt.close()
            plt.cla()
            plt.clf()

        # 繪製 merge bbox 版本
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        fig.canvas.manager.set_window_title("多尺度預測結果_結合 bbox 版")
        fig.suptitle('Merge BBox result', fontsize=12)
        for i, axi in enumerate(ax.flat):
            # i runs from 0 to (nrows*ncols-1)
            # axi is equivalent with ax[rowid][colid]
            img = self.with_merged_bbox_multi_images[i]
            axi.imshow(img, alpha=1.0, cmap=plt.get_cmap('gray'))
            # get indices of row/column
            rowid = i // ncols
            colid = i % ncols
            # 將 行/列索引 寫為軸的標題以供識別
            # axi.set_title("Row:" + str(rowid) + ", Col:" + str(colid))
            if self.auto_thres:
                # 使用動態thres 需要印出更詳細數值
                cur_scale = self.scale_list[i]
                _percentile = str(self.percent_pick_dict[cur_scale])  # 使用的百分位數
                _thres_P = str(self.percent_pick_predictP_dict[cur_scale])  # 百分位數對應的閥值
                ss_percentile = "{:.0f}".format(Decimal(str(_percentile)))
                ss_thres_P = "{:E}".format(Decimal(str(_thres_P)))
                axi.set_title(f"Scale: {str(cur_scale)}, percentile:({ss_percentile})\n={ss_thres_P}")
            else:
                axi.set_title(f"Scale: {str(self.scale_list[i])}")
        plt.tight_layout()
        plt.savefig(Path(self.save_folder).joinpath(f'{self.cursor_img_name}_detection_merge_ver.png'))
        if showit:
            pass
        else:
            plt.close()
            plt.cla()
            plt.clf()

        ###  繪製 '優化過的' merge bbox 版本
        try:
            self.with_BEST_merged_bbox_and_corresponding_img
        except AttributeError as err:
            print("*" * 20)
            print("請先執行 filte_bad_merged_result()，再執行 plot_result(), 錯誤訊息如下:")
            print(err)
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        fig.canvas.manager.set_window_title("多尺度預測結果_篩選過 bbox 的版本")
        fig.suptitle('Best Merge BBox result', fontsize=12)
        # --
        _W_ = self.with_BEST_merged_bbox_and_corresponding_img  # 變數變換
        # 把 _W_，本來是 key 取值的方式轉成 0~n
        _img_keyV = [_ for _ in self.scale_list]
        for i, axi in enumerate(ax.flat):
            # i runs from 0 to (nrows*ncols-1)
            # axi is equivalent with ax[rowid][colid]
            _img_keyName = _img_keyV[i]
            img = self.with_BEST_merged_bbox_and_corresponding_img[_img_keyName]['img']
            opti_bbox = self.with_BEST_merged_bbox_and_corresponding_img[_img_keyName]['bbox']
            image_have_bbox = img.copy()
            for xyxy in opti_bbox:
                cv2.rectangle(image_have_bbox, xyxy[0:2], xyxy[2:], color=(0, 255, 0), thickness=self.thick)

            axi.imshow(image_have_bbox, alpha=1.0, cmap=plt.get_cmap('gray'))
            if self.auto_thres:
                # 使用動態thres 需要印出更詳細數值
                cur_scale = self.scale_list[i]
                _percentile = str(self.percent_pick_dict[cur_scale])  # 使用的百分位數
                _thres_P = str(self.percent_pick_predictP_dict[cur_scale])  # 百分位數對應的閥值
                ss_percentile = "{:.0f}".format(Decimal(str(_percentile)))
                ss_thres_P = "{:E}".format(Decimal(str(_thres_P)))
                axi.set_title(f"Scale: {str(cur_scale)}, percentile:({ss_percentile})\n={ss_thres_P}")
            else:
                axi.set_title(f"Scale: {str(self.scale_list[i])}")
        plt.tight_layout()
        plt.savefig(Path(self.save_folder).joinpath(f'{self.cursor_img_name}_Best_merge_ver.png'))

        if showit:
            pass
        else:
            plt.close()
            plt.cla()
            plt.clf()

        # 總是關閉
        plt.close()
        plt.cla()
        plt.clf()

        # 將 過濾後的 bbox 位置存到txt
        self.write_bbox_to_txt()

    @staticmethod
    def xyxy2yolo(w, h, x1, y1, x2, y2):

        yl_w = abs(x1 - x2) / w
        yl_h = abs(y1 - y2) / h
        yl_x = (x1 + abs(x1 - x2)/2) / w
        yl_y = (y1 + abs(y1 - y2)/2) / h

        return float("{:.18f}".format(yl_x)), \
               float("{:.18f}".format(yl_y)), \
               float("{:.18f}".format(yl_w)), \
               float("{:.18f}".format(yl_h))

    @staticmethod
    def yolo2xyxy(w, h, yolo_x, yolo_y, yolo_w, yolo_h):
        img_w, img_h = int(w*yolo_w), int(h*yolo_h)  # 圖片 寬 高
        half_w, half_h = img_w//2, img_h//2  # 圖片的 半寬 半高

        x1 = int(yolo_x * w) - half_w
        y1 = int(yolo_y * h) - half_h
        x2 = int(yolo_x * w) + half_w
        y2 = int(yolo_y * h) + half_h

        assert x1 < x2 and y1 < y2

        return x1, y1, x2, y2

    def write_bbox_to_txt(self):
        """
        把 方框(bbox_set) 的 bbox 轉成 yolo box 格式，並寫到檔案裡面。
        """
        _ = copy.deepcopy(self.with_BEST_merged_bbox_and_corresponding_img)
        #  後續都針對 _ 做操作

        # bbox 計算成 yolo.
        for scale in self.scale_list:
            res = []
            for x1, y1, x2, y2 in _[scale]['bbox']:
                w, h = _[scale]['img'].shape[1::-1]
                yl_x, yl_y, yl_w, yl_h = self.xyxy2yolo(w, h, x1, y1, x2, y2)
                res.append([yl_x, yl_y, yl_w, yl_h])

            _[scale]['bbox'] = res

        # ['img'] 的 ndarray 設定成 shape.
        for scale in self.scale_list:
            _[scale]['img'] = _[scale]['img'].shape
        _.update({"fn": self.cursor_img_name})  # 寫入檔名
        # 這是印出用的
        # json_object = json.dumps(_, indent=4)
        save_path = self.save_folder
        fn = self.cursor_img_name
        with open(Path(save_path).joinpath(f"{fn}_merged_best_bbox.txt"), mode='w', encoding='utf-8') as f:
            json.dump(_, f, indent=4)

    def filte_bad_merged_result(self, tolerance=33/40):
        """
        過濾掉merge 完很爛的 bbox

        @param tolerance (float):
            bbox 可以允許的極限比，每個 bbox之 短邊:長邊 之 比值 要大於 tolerance，
            才會將他回傳。

        @return:
            dict( [{'bbox':[], 'img': np.ndarray}, ...] )
        """
        assert self.with_merged_bbox_and_corresponding_img is not None  # 請先執行 multiscale_prediction

        self.with_BEST_merged_bbox_and_corresponding_img = dict().fromkeys(self.scale_list, None)

        for sc in self.scale_list:
            #
            bbox_tati = self.with_merged_bbox_and_corresponding_img[sc]['bbox']
            tar_img = self.with_merged_bbox_and_corresponding_img[sc]['img'].copy()
            # build 答案
            self.with_BEST_merged_bbox_and_corresponding_img[sc] = dict({"bbox": [], "img":  tar_img})
            if len(bbox_tati) == 0:
                continue
            else:
                for box in bbox_tati:
                    _w, _h = abs(box[0]-box[2]), abs(box[1]-box[3])  # abs(x1-x2), abs(y1-y2)
                    big, small = _w, _h
                    if _h > _w:
                        big, small = _h, _w
                    if (small/big) >= tolerance:
                        self.with_BEST_merged_bbox_and_corresponding_img[sc]['bbox'].append(box)
                    else:
                        pass
    @staticmethod
    def xyxyBox_IOU(w, h, xyxy1, xyxy2):
        x1,  y1,  x2,  y2 = xyxy1
        x1p, y1p, x2p, y2p = xyxy2
        assert x1 < x2 and y1 < y2
        assert x1p < x2p and y1p < y2p
        x_type1_in_bound = (x1 < x1p < x2) or (x1 < x2p < x2)  # 交錯
        x_type2_in_bound = (x1 < x1p < x2p < x2) or (x1p < x1 < x2 < x2p)  # 包覆
        y_type1_in_bound = (y1 < y1p < y2) or (y1 < y2p < y2)  # 交錯
        y_type2_in_bound = (y1 < y1p < y2p < y2) or (y1p < y1 < y2 < y2p)  # 包覆

        x_BOUND = x_type1_in_bound or x_type2_in_bound
        y_BOUND = y_type1_in_bound or y_type2_in_bound

        IOU_RESULT = 0
        if x_BOUND and y_BOUND:
            # calc IOU
            # TODO
            pass

        return IOU_RESULT

if __name__ == "__main__":

    origin_detector = Detector(weight="./50weight.pt", save_folder=LOG_SAVE_FOLDER,
             net=qrcnn_model.QRCode_CNN())


    #fname_list = glob.glob("./data_clean/the_real593/*.*")
    fname_list = glob.glob("./data_clean/NOT_IN_the_real593_DATASET/*.*")
    cnt = 0
    import random
    pbar = tqdm(fname_list)
    #for idx in range(0, len(fname_list)):
    for fn in pbar:
        pbar.set_description("process: %s" % fn)
        # rand_pick = random.randint(0, len(fname_list)-1)  # for random
        img_path = fn # fname_list[idx]
        # img_path = fname_list[rand_pick] # for random
        origin_detector.multiscale_prediction(img_path)
        origin_detector.filte_bad_merged_result()  # 把長寬比不足篩掉，篩完後要處理  origin_detector.with_BEST_merged_bbox_and_corresponding_img
        origin_detector.plot_result(showit=False)

        # cnt += 1
        # if cnt > 20:
        #     break

