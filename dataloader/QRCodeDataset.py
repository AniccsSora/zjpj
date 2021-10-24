from torch.utils.data import Dataset
import glob
from os.path import join as pjoin
import cv2


class QRCodeDataset(Dataset):
    def __init__(self, annotations_dir, img_dir, predefined_class_file):
        self.annotations_dir = annotations_dir
        self.img_dir = img_dir
        self.predefined_class_file = predefined_class_file

        self.defined_class = self.read_predefine_class()

        self.img_paths = [fname for fname in glob.glob(pjoin(img_dir, '*'))]
        self.img_labels = self.read_yolo_labels()

        assert len(self.img_paths) == len(self.img_labels)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image = cv2.imread(self.img_paths[idx], cv2.COLOR_BGR2GRAY)
        return self.img_labels[idx], image

    def read_yolo_labels(self):
        # """ 根據 annotations_dir 內的資料 load labels資料 """

        def read_txt(fpth):
            # 讀取單個 yolo label file
            res = []

            with open(fpth) as f:
                lines = f.readlines()
                lines = [_.rstrip() for _ in lines]

                for bbox in lines:
                    c, x, y, w, h = [_ for _ in bbox.split(' ')]
                    res.append((c, x, y, w, h))
            return res

        # ====================================
        image_labels = []
        for lb_path in glob.glob(pjoin(self.annotations_dir, '*')):
            image_labels += [read_txt(lb_path)]

        return image_labels

    def read_predefine_class(self):
        cdict = dict()
        with open(self.predefined_class_file) as f:
            lines = f.readlines()
            lines = [_.rstrip() for _ in lines]
            for idx, cname in enumerate(lines):
                cdict[idx] = cname
        return cdict


if __name__ == "__main__":
    qr_code_dataset = QRCodeDataset(annotations_dir="../data/paper_qr_label_yolo",
                                    img_dir="../data/paper_qr",
                                    predefined_class_file="../data/predefined_classes.txt")


