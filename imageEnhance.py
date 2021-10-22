import xml.etree.ElementTree as ET
from misc.voc_xml_parser import VocParser
import imageio
import os
import misc.F as util
from imgaug.augmentables.bbs import BoundingBoxesOnImage
import imgaug.augmenters as iaa
from random import randint, uniform
import imgaug

class ImEnhance:
    """

    """
    def __init__(self, random_order=True, random_pick=False, pick=2):
        self.random_order = random_order
        self.random_pick = random_pick
        self.pick = pick
        self._current_w = None
        self._current_h = None
        # enhance 的 seq
        self.enhance_event = None

    def _regen_event(self):
        _887 = self.pick  # useless
        self.enhance_event = [
            iaa.Affine(name='rotation', rotate=randint(-10, 10)),  # 隨機旋轉 A~B
            iaa.Affine(shear=(-16, 16)),  # Shear, 剪力 ,Shear mapping
            iaa.Affine(translate_percent={"x": -0.20}, mode=imgaug.ALL, cval=(0, 255)),
            iaa.Multiply((0.5, 1.5)),  # 對整張圖隨機乘 A~B 之間的值
            iaa.MultiplyElementwise((0.5, 1.5)),  # element-wise 隨機乘 A~B 之間的值
            iaa.ReplaceElementwise(uniform(0, 0.1), [randint(0, 255), randint(0, 255)]),  # 椒鹽雜訊 亂用版本
            iaa.Cutout(nb_iterations=(1, randint(2, 10)), size=0.05, squared=False),  # 隨機 cutout，
            iaa.Dropout(p=(0, 0.05)),  # 對 0<= p <= 0.1 數量的像素版分比，將pixel設定成黑色
            iaa.JpegCompression(compression=(50, 99)),  # randomly and uniformly sampled per image
            iaa.Affine(
                translate_px={"x": randint(-5, 5), "y": randint(-5, 5)},  # 平行移動
                scale=(uniform(.95, 1), uniform(.95, 1)),  # 分別針對 xy 軸做拉伸
            )
        ]


    def augument(self, fn, seq=None):
        self._regen_event()
        image = imageio.imread(fn.get_image_path())  # os.path.join(args.img_path,
        self._current_w, self._current_h = image.shape[1::-1]
        bboxes = util.get_BoundingBoxes(fn.voc_xyxy(have_label=False))  #  box list -> 物件 boxes
        bbs = BoundingBoxesOnImage(bboxes, shape=image.shape)  # 畫框在圖上
        if seq is None:
            seq = self.get_seq()
        # 取得這次隨機的角度
        seq_param = {"rotation": seq.find_augmenters_by_name('rotation')[0].rotate.value,
                     }
        return seq(image=image, bounding_boxes=bbs), seq_param

    def get_seq(self):
        """ 取得強化用的 sequence。
        Args:
            (先判定)random_pick: 是否要隨機(預設=否)

            random_order: sequential 使用隨機順序。(預設=否)

            pick: 當 random_pick=True 時有用。為隨機取樣的數量，設定 -1 時隨機選取。

            event: 為 list，sequence的種類。

            image_ref_size: 該圖片大小，預設不使用，目的是有規則的設定 translate_px。

        return:
            強化用的 sequence

        assert:
            pick 高於可選的 event 總數。


        """
        if self.random_pick == True:
            raise NotImplementedError("因為要採集旋轉資訊，故不可以隨機挑。")

        if not self.random_pick:
            if self.random_order:
                return iaa.Sequential(self.enhance_event, random_order=self.random_order)
            else:
                return iaa.Sequential(self.enhance_event)
        else:
            assert len(self.enhance_event) >= self.pick
            if self.pick < 0:
                if randint(1, 100) < 10:  # 10% 機率 使用隨機序列。
                    return iaa.Sequential(self.enhance_event, random_order=True)
                else:
                    return iaa.SomeOf(randint(1, len(self.enhance_event)), self.enhance_event)
            else:
                return iaa.SomeOf(self.pick, self.enhance_event)


if __name__ == "__main__":
    # xml 的 labeling data
    # xml_dlist = VocParser(args.xml_path).get_dlist()

    enhancer = ImEnhance()