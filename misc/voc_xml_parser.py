import xml.etree.ElementTree as ET
import glob
import os


class VocParser:
    """
    path: xml folder.
    """
    def __init__(self, path):

        if not os.path.exists(path):  # 確保 xml 存在。
            raise ValueError("路徑 {} 不存在".format(path))

        self.paths = glob.glob(os.path.join(path.rstrip(os.path.sep))+os.path.sep+'*.xml')

    def get_dlist(self):
        """
        return: list(), all of each image bbox coordi and other params.
        """
        tokens = []
        for _ in self:
            tokens.append(_)
        return tokens


    def __getitem__(self, idx):
        tree = ET.parse(self.paths[idx])
        res = dict()
        for _ in tree.findall('path'):
            res.update({'path': _.text})

        for _ in tree.findall('filename'):
            res.update({'filename': _.text})

        for _ in tree.findall('size'):
            res.update({'width': _.find('width').text})
            res.update({'height': _.find('height').text})
            res.update({'depth': _.find('depth').text})
        bndboxs = []
        for _ in tree.findall('object'):
            x0 = _.find('bndbox').find('xmin').text
            y0 = _.find('bndbox').find('ymin').text
            x1 = _.find('bndbox').find('xmax').text
            y1 = _.find('bndbox').find('ymax').text
            bndboxs.append([x0, y0, x1, y1])
        res.update({'bndboxs': bndboxs})

        class NodeElement:
            xml_path = self.paths[idx]
            img_path = res['path']
            filename = res['filename']
            purefname = filename.rsplit('.')[0]
            bndboxs = res['bndboxs']

        return NodeElement



if __name__ == "__main__":
    test1 = False  # findAll() 方法
    test2 = False  # 走訪
    test3 = False  # 迭代 iter() 方法
    text4 = True  # 寫回檔案

    if test1:
        # 使用方法，此目錄下為 voc 格式的 xml 們。
        path = r'D:\ppppppppppp\!Zhang-Jia Project\program\label-qr-code'

        # dlist = dict list
        xml_dlist = VocParser(path).get_dlist()

        print("==================\n")
        # 透過迭代的方式取得每筆資料的訊息
        for label_dict in xml_dlist[:3]:
            for key, item in label_dict.items():
                print(key, item)
            print("==================")
        print('exit')

    if test2:
        # 單獨測試 xml 讀寫
        fname = '../data/label-qr-code/qr_0001.xml'
        tree = ET.parse(fname)

        # 這會得到根結點，要知道下面的元素都需要走訪
        root = tree.getroot()

        # child 的型態為 Element
        for child in root:
            print("ctag:{}, cattrib:{}, ctext:{}".format(child.tag, child.attrib, child.text))

        print('exit')

    if test3:
        fname = '../data/label-qr-code/qr_0001.xml'
        tree = ET.parse(fname)

        root = tree.getroot()

        # 注意 <...>這邊是 text</...>
        # iter 他好像只會迭代該層所有元素(單個)而已
        # iter('xmin') 中的字串跟 xml 有關。
        for xmin in root.iter('xmin'):
            print(xmin.text)

    if text4:
        fname = '../data/label-qr-code/qr_0009.xml'
        tree = ET.parse(fname)

        root = tree.getroot()

        for xmin in root.iter('xmin'):
            new_xmin = int(xmin.text) + 111
            print("修改前: {}".format(xmin.text))
            xmin.text = str(new_xmin) #  修改
            print("修改後: {}".format(xmin.text))

        # 寫回檔案
        tree.write("./修改過的.xml")

        print('exit')
