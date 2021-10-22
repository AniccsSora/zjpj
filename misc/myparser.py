# -*- utf-8 -*-
from glob import glob
import os
from PIL import Image


class YoloLabelReader:

    def __init__(self, label_dir, image_dir, single_label=True):
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.single = single_label
        self._corresponds_image_pth = None
        self._image = None
        self._current_label_list = None

    def __len__(self):
        l = len(glob(self.label_dir))
        i = len(glob(self.image_dir))
        if l != i:
            print(f"warn : label資料夾資料量: {l}, image資料量: {i}.")
        return l

    def __getitem__(self, fn):
        lfn = os.path.join(self.label_dir, fn+'.txt')
        if not os.path.isfile(lfn):
            raise ValueError(f'label path "{lfn}" does not exists!')
        self._corresponds_image_pth = os.path.join(self.image_dir, fn+'.*')
        ifn = glob(self._corresponds_image_pth)
        if ifn is []:
            raise ValueError(f'Corresponds image file "{self._corresponds_image_pth}" does not exists!')

        self._image = Image.open(ifn[0])

        res = []
        try:
            with open(lfn, encoding='utf-8') as fp:
                res = fp.readlines()
        except UnicodeDecodeError as e:
            print(f'{e} -- by -- "{lfn}"')

        self._current_label_list = [_.replace('\n', '') for _ in res]
        return self

    def yolo_xywh(self, have_label: bool):
        if have_label:
            assert have_label is False  # 這邊回傳的要改因為
            return self._current_label_list
        else:
            return [_.split(' ')[1:] for _ in self._current_label_list]

    def voc_xyxy(self, have_label: bool):
        if have_label:
            assert have_label is False  # 這邊回傳的要改因為
            return self._current_label_list
        else:
            w, h = self._image.size
            res = [_.split(' ')[1:] for _ in self._current_label_list]
            new_res = []
            for xc, yc, bw, bh in res:
                xmin = float(xc) * w - (w/2) * float(bw)
                ymin = float(yc) * h - (h/2) * float(bh)
                xmax = float(xc) * w + (w/2) * float(bw)
                ymax = float(yc) * h + (h/2) * float(bh)
                tmp = [round(xmin), round(ymin), round(xmax), round(ymax)]
                new_res.append(tmp)
            return new_res

    def get_image_path(self):
        if glob(self._corresponds_image_pth) is []:
            raise ValueError(f'not found this pattern file "{self._corresponds_image_pth}".')
        return glob(self._corresponds_image_pth)[0]