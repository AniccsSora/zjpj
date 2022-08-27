import torch
import cv2
import matplotlib.pyplot as plt

# yolo weight path
wp = './exp2_補500張e100_arg_high/weights/best.pt'
# verbose = False 把 use cache 訊息遮蔽。
model = torch.hub.load('ultralytics/yolov5', 'custom', wp, verbose=False)


def get_xyxy(img, norm=False):
    #assert isinstance(img, str)
    result = model(img)

    w, h = cv2.imread(img).shape[1::-1]
    xyxypc_res = []
    for xyxys in result.xyxy:
        for xyxy in xyxys:  # 一張圖片的判斷結果
            cpu_xyxy = xyxy.detach().cpu().numpy()
            x1, y1, x2, y2, p, cls_idx = cpu_xyxy
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            if not norm:
                xyxypc_res.append((x1, y1, x2, y2, p, cls_idx))
            else:
                assert x1 / w <= 1
                assert y1 / h <= 1
                assert x2 / w <= 1
                assert y2 / h <= 1
                xyxypc_res.append((x1/w, y1/h, x2/w, y2/h, p, cls_idx))
    # p:機率, c:類別名
    return xyxypc_res


if __name__ == "__main__":
    imgs = [r"D:\git-repo\zjpj\data\raw_qr\qr_0025.jpg"]
    # r'D:\Git\yolov5\data\images\504.png'
    for img in imgs:
        cv_im = cv2.imread(img)
        res = get_xyxy(cv_im)  # get_xyxy: 回傳 list，根據 bbox數量，裡面放置xyxypc資料
        for res_single in res:
            x1,y1,x2,y2,p,c = res_single
            cv2.rectangle(cv_im, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.imshow('result', cv_im)
        cv2.waitKey(0)
