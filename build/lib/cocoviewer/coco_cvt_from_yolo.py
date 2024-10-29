import os
import cv2
import json
import imagesize
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import argparse

def train_test_val_split_by_files(img_paths, yolo_root):
    # 根据文件 train.txt, val.txt, test.txt（里面写的都是对应集合的图片名字） 来定义训练集、验证集和测试集
    phases = ['train', 'val', 'test']
    img_split = []
    for p in phases:
        define_path = os.path.join(yolo_root, f'{p}.txt')
        print(f'Read {p} dataset definition from {define_path}')
        assert os.path.exists(define_path)
        with open(define_path, 'r') as f:
            img_paths = f.readlines()
            # img_paths = [os.path.split(img_path.strip())[1] for img_path in img_paths]  # NOTE 取消这句备注可以读取绝对地址。
            img_split.append(img_paths)
    return img_split[0], img_split[1], img_split[2]


def yolo2coco():
    parser = argparse.ArgumentParser()
    parser.add_argument('yolo_root', default='./', type=str,
                        help="root path of images and labels, include ./images and ./labels and classes.txt")
    parser.add_argument('out_json_path', type=str, default='./train.json',
                        help="if not split the dataset, give a path to a json file")
    arg = parser.parse_args()
    root_path = arg.yolo_root
    print("Loading data from ", root_path)

    assert os.path.exists(root_path)
    originLabelsDir = os.path.join(root_path, 'labels')
    originImagesDir = os.path.join(root_path, 'images')
    with open(os.path.join(root_path, 'classes.txt')) as f:
        classes = f.read().strip().split()
    # images dir name
    indexes = os.listdir(originImagesDir)

    dataset = {'categories': [], 'annotations': [], 'images': []}
    for i, cls in enumerate(classes, 0):
        dataset['categories'].append(
            {'id': i, 'name': cls, 'supercategory': 'mark'})

    # 标注的id
    ann_id_cnt = 0
    for k, index in enumerate(tqdm(indexes)):
        # 获取标签 txt 文件名称，支持 png jpg 格式的图片。
        txtFile = index.replace('images', 'txt').replace(
            '.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
        # 读取图像的宽和高
        width, height = imagesize.get(
            os.path.join(root_path, 'images/') + index)

        if not os.path.exists(os.path.join(originLabelsDir, txtFile)):
            # 如没标签，跳过，只保留图片信息。
            continue
        dataset['images'].append({'file_name': index,
                                  'id': k,
                                  'width': width,
                                  'height': height})

        with open(os.path.join(originLabelsDir, txtFile), 'r') as fr:
            labelList = fr.readlines()
            for label in labelList:
                label = label.strip().split()
                x = float(label[1])
                y = float(label[2])
                w = float(label[3])
                h = float(label[4])

                # convert x,y,w,h to x1,y1,x2,y2
                H, W = height, width
                x1 = (x - w / 2) * W
                y1 = (y - h / 2) * H
                x2 = (x + w / 2) * W
                y2 = (y + h / 2) * H
                # 标签序号从0开始计算, coco2017数据集标号混乱，不管它了。
                cls_id = int(label[0])
                box_width = max(0, x2 - x1)
                box_height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': box_width * box_height,
                    'bbox': [x1, y1, box_width, box_height],
                    'category_id': cls_id,
                    'id': ann_id_cnt,
                    'image_id': k,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                ann_id_cnt += 1

    # 保存结果
    folder = os.path.join(root_path, 'annotations')
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(arg.out_json_path, 'w') as f:
        json.dump(dataset, f)
        print('Save annotation to ', arg.out_json_path)
