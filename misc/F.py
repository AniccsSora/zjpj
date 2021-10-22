import misc.logger as logger
from imgaug.augmentables.bbs import BoundingBox
import numpy as np
import os
import shutil
import xml.etree.ElementTree as ET
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imageio
import imgaug
from PIL import Image
from glob import glob
import os
import random
import imageio
import math


def dataset_split(dir, train=0.7, valid=0.3, image_format=['png', 'jpeg', 'jpg'],
                  labels_format=['txt'], train_dir='train', valid_dir='valid',
                  args=None, _logger=None):
    # train
    train_subset_dir = os.path.join(dir, train_dir)
    ensure_folder(train_subset_dir, remake=True)
    train_image_dir = os.path.join(train_subset_dir, "images")
    train_labels_dir =os.path.join(train_subset_dir, "labels")
    ensure_folder(train_image_dir)
    ensure_folder(train_labels_dir)
    # vaild
    valid_subset_dir = os.path.join(dir, valid_dir)
    ensure_folder(valid_subset_dir, remake=True)
    valid_image_dir = os.path.join(valid_subset_dir, "images")
    valid_labels_dir = os.path.join(valid_subset_dir, "labels")
    ensure_folder(valid_image_dir)
    ensure_folder(valid_labels_dir)

    flatten = lambda t: [item for sublist in t for item in sublist]

    _tmp = flatten([glob(os.path.join(dir, f'*.{sn}')) for sn in image_format])
    all_img_fn = [_.replace('/', ' ')
                  .replace('\\', ' ')
                  .replace('.', ' ')
                  .split(' ')[-2] for _ in _tmp]

    all_number = len(all_img_fn)
    train_number = int(math.floor(all_number * train))
    valid_number = int(math.ceil(all_number * valid))

    if train_number + valid_number != all_number:
        print(f'train_number:{train_number} + valid_number{valid_number} != '
              f'all_number:{all_number}, 並不相等。')
        _offset = (train_number+valid_number-all_number)
        valid_number = int(valid_number - _offset)
        print("調整為:")
        print("train_number:", train_number)
        print("valid_number:", valid_number)

    if args is not None and args.verbose:
        print(f"image 使用副檔名: {image_format}")
        print(f"label 使用副檔名: {labels_format}")

    random.shuffle(all_img_fn)
    train_sub = all_img_fn[0:train_number]
    valid_sub = all_img_fn[train_number:]

    if args is not None and args.verbose:
        print("train {}筆: {}".format(len(train_sub), train_sub))
        print("valid {}筆: {}".format(len(valid_sub), valid_sub))
    else:
        print("train {}筆".format(len(train_sub)))
        print("valid {}筆".format(len(valid_sub)))
    print("共 {} 筆.".format(len(train_sub)+len(valid_sub)))

    print_logger = None
    if _logger is not None:
        print_logger = _logger.info
    else:
        print_logger = NOTHING

    # moving train subset
    for fn in train_sub:
        for sub_n in image_format:
            target_f = fn + f'.{sub_n}'
            full_path = os.path.join(dir, target_f)
            if os.path.isfile(full_path):
                shutil.copy2(src=full_path, dst=train_image_dir)
                print_logger("'{}' -- copy2 --> '{}'".format(full_path, train_image_dir))
        for sub_n in labels_format:
            target_f = fn + f'.{sub_n}'
            full_path = os.path.join(dir, target_f)
            if os.path.isfile(full_path):
                shutil.copy2(src=full_path, dst=train_labels_dir)
                print_logger("'{}' -- copy2 --> '{}'".format(full_path, train_labels_dir))
    print("已將 train 子集 複製至 {}".format(os.path.join(dir, train_dir)))
    # valid image
    for fn in valid_sub:
        for sub_n in image_format:
            target_f = fn + f'.{sub_n}'
            full_path = os.path.join(dir, target_f)
            if os.path.isfile(full_path):
                shutil.copy2(src=full_path, dst=valid_image_dir)
                print_logger("'{}' -- copy2 --> '{}'".format(full_path, valid_image_dir))
        for sub_n in labels_format:
            target_f = fn + f'.{sub_n}'
            full_path = os.path.join(dir, target_f)
            if os.path.isfile(full_path):
                shutil.copy2(src=full_path, dst=valid_labels_dir)
                print_logger("'{}' -- copy2 --> '{}'".format(full_path, valid_labels_dir))
    print("已將 valid 子集 複製至 {}".format(os.path.join(dir, valid_dir)))


def NOTHING(*args):
    pass

def get_image_filenames(dir, format=['png', 'jpeg', 'jpg'], full_path=True):
    flatten = lambda dl: [e for sub in dl for e in sub]
    res = flatten([glob(os.path.join(dir, f'*.{fm}')) for fm in format])
    if full_path:
        return res
    else:
        return [_.replace('\\', ' ').replace('/', ' ').replace('.', ' ').split(' ')[-2] for _ in res]


def img_folder_chk(dir, format=['png', 'jpeg', 'jpg'], logger=None):
    noi = len(glob(os.path.join(dir, '*.*')))
    fname = [glob(os.path.join(dir, f'*.{fm}')) for fm in format ]

    for imtype in fname:
        for img_p in imtype:
            Image.open(img_p).convert('RGB').save(img_p)
    if logger is not None:
        logger.info("image folder check successfully!")


def show_bbox_on_image(bboxs, img_path, mode='xyxy', save=True):

    image = imageio.imread(img_path)

    xyxy = []
    if mode.lower()=='xyxy':
        for x0, y0, x1, y1 in bboxs:
            xyxy.append(BoundingBox(int(x0), int(y0), int(x1), int(y1)))
    elif mode.lower()=='xywh':
        w, h = image.shape[1::-1]
        for xc, yc, bw, bh in bboxs:
            xmin = float(xc) * w - (w/2) * float(bw)
            ymin = float(yc) * h - (h/2) * float(bh)
            xmax = float(xc) * w + (w/2) * float(bw)
            ymax = float(yc) * h + (h/2) * float(bh)
            xyxy.append(BoundingBox(int(xmin), int(ymin), int(xmax), int(ymax)))
    else:
        raise ValueError(f'Not support mode "{mode}". Consider use xyxy(voc format) or xywh(yolo format).')
    bbs = BoundingBoxesOnImage(xyxy, shape=image.shape)

    bbs.draw_on_image(image, size=2)
    imgaug.imshow(bbs.draw_on_image(image, size=2))
    if save:
        imageio.imsave(os.path.join(os.getcwd(), 'test_show.jpg'), bbs.draw_on_image(image, size=2))


def show_bbox_on_image_xmlver(xml_path, img_path, save=False):

    image = imageio.imread(img_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    xyxy = []

    for _ in tree.findall('object'):
        x0 = _.find('bndbox').find('xmin').text
        y0 = _.find('bndbox').find('ymin').text
        x1 = _.find('bndbox').find('xmax').text
        y1 = _.find('bndbox').find('ymax').text
        xyxy.append(BoundingBox(int(x0), int(y0), int(x1), int(y1)))

    bbs = BoundingBoxesOnImage(xyxy, shape=image.shape)

    bbs.draw_on_image(image, size=2)
    imgaug.imshow(bbs.draw_on_image(image, size=2))
    if save:
        imageio.imsave(os.path.join(os.getcwd(), 'test_show.jpg'), bbs.draw_on_image(image, size=2))

def rewrite_xyxy2xml(xyxy, xml_path, rewrite_dir, rewrite_fname, label=['xmin', 'ymin', 'xmax', 'ymax']):
    """
    注意這目前只適用於單一物件的偵測使用。多目標的他不會管誰是誰就直接取代。

    將某一個xml的所有 object 的 bbox 都帶換掉

    # xyxy: default: ['xmin', 'ymin', 'xmax', 'ymax']。

    # label: !!!DEPENDENT on !!! xml label format.

    # 處理節點範例如下
    <object>
		<name>QRCode</name>
		<pose>Unspecified</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>352</xmin>
			<ymin>321</ymin>
			<xmax>478</xmax>
			<ymax>501</ymax>
		</bndbox>
	</object>
	"""
    # DEPENDENT!!! string
    def get_value_object_node(xmin, ymin, xmax, ymax):
        n_object = ET.Element('object')
        name = ET.SubElement(n_object, 'name')
        pose = ET.SubElement(n_object, 'pose')
        truncated = ET.SubElement(n_object, 'truncated')
        difficult = ET.SubElement(n_object, 'difficult')
        bndbox = ET.SubElement(n_object, 'bndbox')
        xmin_n = ET.SubElement(bndbox, 'xmin')
        ymin_n = ET.SubElement(bndbox, 'ymin')
        xmax_n = ET.SubElement(bndbox, 'xmax')
        ymax_n = ET.SubElement(bndbox, 'ymax')

        name.text = 'QRCode'
        pose.text = 'Unspecified'
        truncated.text = '0'
        difficult.text = '0'
        xmin_n.text = '0'#str(xmin)
        ymin_n.text = '0'#str(ymin)
        xmax_n.text = '0'#str(xmax)
        ymax_n.text = '0'#str(ymax)
        return n_object

    ensure_folder(rewrite_dir)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for object_node in root.findall('object'):
        root.remove(object_node)

    for xmin, ymin, xmax, ymax in xyxy:
        nn = get_value_object_node(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        root.append(nn)  # 在 最後一個 node 後面添加新的節點

    tree.write(os.path.join(rewrite_dir, rewrite_fname))

def Deprecated_rewrite_xyxy2xml(xyxy, xml_path, rewrite_dir, rewrite_fname, label=['xmin', 'ymin', 'xmax', 'ymax']):
    """
    注意這目前只適用於單一物件的偵測使用。多目標的他不會管誰是誰就直接取代。

    將某一個xml的所有 object 的 bbox 都帶換掉

    # xyxy: default: ['xmin', 'ymin', 'xmax', 'ymax']。

    # label: !!!DEPENDENT on !!! xml label format.
    """
    ensure_folder(rewrite_dir)
    tree = ET.parse(xml_path)
    root = tree.getroot()



    for label_idx, clabel in enumerate(label):  # 迭代 label參數
        for idx, val in enumerate(root.iter(clabel)):  # 迭代 xml 標籤
            val.text = str(xyxy[idx][label_idx])  # 修改

    tree.write(os.path.join(rewrite_dir, rewrite_fname))


def get_BoundingBoxes(bboxeslist):
    assert np.array(bboxeslist).ndim == 2

    res = []
    for bbox in bboxeslist:
        # x1,x2, y1,y2 ,  VOC Pascal is (x1,y1,x2,y2)

        x1, x2, y1, y2 = float(bbox[0]), float(bbox[2]), float(bbox[1]), float(bbox[3])
        res.append(BoundingBox(x1=x1, x2=x2, y1=y1, y2=y2, label=None))
    return res


def ensure_folder(folder_path, remake=False, logger=None):
    """
    確保某個資料夾必定存在，因為會重新建立。
    """
    if os.path.isdir(folder_path):
        if not remake:
            if logger is not None:
                logger.info("已經存在 {} 不須建立".format(folder_path))
            return
        else:
            if logger is not None:
                logger.warning("已經存在 {}".format(folder_path))
            shutil.rmtree(folder_path)
            if logger is not None:
                logger.warning("已刪除 {} 資料夾".format(folder_path))
            os.makedirs(folder_path, 0o755)
            if logger is not None:
                logger.info("已重新建立 {} 資料夾".format(folder_path))
    else:
        os.makedirs(folder_path, 0o755)
        if logger is not None:
            logger.info("已經建立 {} 資料夾".format(folder_path))


def xyxy2xywh(xyxy, w, h):
    assert len(xyxy) == 4
    xywh = []
    xmin, ymin, xmax, ymax = xyxy[0], xyxy[1], xyxy[2], xyxy[3]

    xc = (xmin + xmax) / 2 * (1 / w)
    yc = (ymin + ymax) / 2 * (1 / h)
    ww = (xmax - xmin) / w
    hh = (ymax - ymin) / h

    return [xc, yc, ww, hh]


def write_label_and_image2(img_save_path, image, fn, bbs, logger, allow_negative=False):
    imageio.imsave(img_save_path, image)
    w, h = image.shape[1::-1]
    logger.info(f"Image '{img_save_path}' saved.")
    tmp_xywh = []
    for bb in bbs:
        xywh = xyxy2xywh([bb.x1_int, bb.y1_int, bb.x2_int, bb.y2_int], w, h)
        if allow_negative or bbox_not_negative(xywh):
            xywh = ['{:.6f}'.format(n) for n in xywh]
            tmp_xywh.append("0 " + " ".join(xywh)+'\n')
        else:
            pass

    with open(os.path.join(logger.get_log_dir(), fn+'.txt'), 'w') as fp:
        fp.writelines(tmp_xywh)


def bbox_not_negative(bbox):
    for val in bbox:
        if val < 0:
            return False
    return True

def command_gen(aug_number, **args):
    command = "python train.py"
    for key, val in args.items():
        if key == 'name':
            val = val + str(aug_number) + '_e{}'.format(args['epochs'])
        command = command + ' --{} '.format(key.replace('_', '-')) + str(val)
    return command


def check_line_be_than(line, val=0, sep=' '):
    """
    :param line: yolo 格式的 label，一個row。
    :param val: 一般來說都要 > 0。
    :param sep: 用什麼符號分隔，一般來說是 空白
    :return: 告訴你此行有無負號。
    """
    line = line.rstrip('\n')
    bbox = [float(_) for _ in line.split(sep)[1:]]  # 略過首項(class)

    for i, b in enumerate(bbox):
        if b >= val:
            bbox[i] = True
        else:
            bbox[i] = False

    return np.all(bbox)


def just_get_file_name(file_name):
    """
    :param file_name: 完整路徑的路徑，通常代表一個檔案，只是是完整路徑。
    :return: 返還檔案名(包含副檔名)。
    """
    sep = os.path.sep
    return file_name[file_name.rfind(sep)+1:]


def remove_negative_bbox(label_pth, re_dir=False, file_type='txt', fn_ptn='qr*', sep=' '):
    save_path = label_pth
    total_cnt = 0
    if re_dir:
        save_path = os.path.join(label_pth, 'remake_labels')
        ensure_folder(save_path)
    fnumber = len(glob(os.path.join(label_pth, fn_ptn+'.'+file_type)))
    for fpfn in glob(os.path.join(label_pth, fn_ptn+'.'+file_type)):
        new_lines = []
        remove_cnt = 0
        with open(fpfn, mode='r') as fp:
            lines = fp.readlines()
            for line in lines:
                if check_line_be_than(line):  # 是否均 >= 0。
                    new_lines.append(line)
                else:
                    remove_cnt += 1

        fn = os.path.join(save_path, just_get_file_name(fpfn))
        with open(fn, mode='w') as fp:
            fp.writelines(new_lines)
            print(f"rewrite OK (remove {remove_cnt} line(s)): {fn}")

        if remove_cnt > 0:
            total_cnt += 1

    print(f"\ntotal modified file: {total_cnt} file(s) / {fnumber} files.\n")


if __name__ == "__main__":
    remove_negative_bbox(r"D:\git-repo\ZhangJiaPJ\log_XD\20210520_2206")

    # print(command_gen(aug_number=10, epochs=300, name='aug',
    #             data="train_my_qr.yaml",
    #             cfg="yolov5s.yaml",
    #             batch_size=96))
    # python train.py --epochs 300 --name aug100_e300 --data train_my_qr.yaml --cfg yolov5s.yaml --batch-size 96

    # python train.py --epochs 600 --name aug1000_e600 --data train_my_qr.yaml --cfg yolov5s.yaml --batch-size 96


