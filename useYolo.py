import torch
import cv2

# yolo weight path
wp = r'D:\Git\yolov5\runs\train\exp2_補500張e100_arg_high\weights\best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', wp)

def get_xyxy(img):
    assert isinstance(img, str)
    result = model(img)

    xyxypc_res = []
    for xyxys in result.xyxy:
        for xyxy in xyxys:  # 一張圖片的判斷結果
            cpu_xyxy = xyxy.detach().cpu().numpy()
            x1, y1, x2, y2, p, cls_idx = cpu_xyxy
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            xyxypc_res.append((x1, y1, x2, y2, p, cls_idx))
    # p:機率, c:類別名
    return xyxypc_res


if __name__ == "__main__":
    imgs = [r'D:\Git\yolov5\data\images\502.png', r'D:\Git\yolov5\data\images\504.png']
    for img in imgs:
        cv_im = cv2.imread(img)
        res = get_xyxy(img)  # get_xyxy: 回傳 list，根據 bbox數量，裡面放置xyxypc資料
        # for res_single in res:
        #     x1,y1,x2,y2,p,c = res_single
        #     cv2.rectangle(cv_im, (x1, y1), (x2, y2), (0, 255, 0), 3)
        # cv2.imshow('result', cv_im)
        # cv2.waitKey(0)
