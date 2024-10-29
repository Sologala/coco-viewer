import pycocotools
import pprint
from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from sahi.utils.coco import Coco
from sahi.utils.file import save_json

from tqdm import tqdm
import json


def plotstatiscEntry():
    parser = argparse.ArgumentParser()
    parser.add_argument("ann_path")
    parser.add_argument("--save", default=False, help="flag to save plot")
    args = parser.parse_args()
    annFile = args.ann_path
    plotstatisc(annFile, args.save)


def plotstatisc(annFile, save):
    # Initialize COCO API for instance annotations
    coco = COCO(annFile)

# Print statistics
    categories = coco.loadCats(coco.getCatIds())


# Get annotations
    catIds = coco.getCatIds()
    print(catIds)
    annIds = coco.getAnnIds()
    anns = coco.loadAnns(annIds)

# Calculate bbox width, height, and aspect ratio
    bbox_areas = np.array([ann['bbox'][2] * ann['bbox'][3] for ann in anns])
    bbox_widths = np.array([ann['bbox'][2] for ann in anns])
    bbox_heights = np.array([ann['bbox'][3] for ann in anns])

    mask = bbox_heights != 0
    if np.sum(mask) == 0:
        aspect_ratios = np.zeros_like(bbox_areas)
    else:
        bbox_heights_no_zero = bbox_heights[bbox_heights != 0]
        bbox_width_no_zero = bbox_widths[bbox_heights != 0]
        aspect_ratios = bbox_width_no_zero / bbox_heights_no_zero

# Calculate bbox centers
    centers_x = np.array([ann['bbox'][0] + ann['bbox'][2]/2 for ann in anns])
    centers_y = np.array([ann['bbox'][1] + ann['bbox'][3]/2 for ann in anns])

# Plot distributions
    plt.figure(figsize=(16, 10))

    plt.subplot(2, 1, 1)
    annids = [ann['category_id'] for ann in anns]
    plt.hist(annids, bins=len(categories))
    plt.xticks(range(1, len(categories)+1),
               [cat['name'] for cat in categories], rotation=0)
    plt.title(("images num: {} , ".format(len(coco.getImgIds()))) +
              ("cat num: {}".format(len(categories))) + ("ann num: {}".format(len(annids))))

    plt.subplot(2, 3, 4)
    plt.hist(bbox_widths, bins=50)
    plt.title('Bbox Width Distribution')

    plt.subplot(2, 3, 5)
    plt.hist(bbox_heights, bins=50)
    plt.title('Bbox Height Distribution')

    plt.subplot(2, 3, 6)
    plt.hist(aspect_ratios, bins=50)
    plt.title('Aspect Ratio Distribution')
    if (save):
        coco_json_plot_path = annFile.replace(".json", "_statistic_.png")
        plt.savefig(coco_json_plot_path, bbox_inches='tight')
    plt.show()

    print("Num of images: {}".format(len(coco.getImgIds())))
    print("Num of categories: {}".format(len(categories)))
    nphist, bin_edges = np.histogram(
        annids, bins=range(0, len(categories) + 1))  # 获取直方图计数
    print(nphist)
    for category in categories:
        print("Category {}: {}".format(category['id'], category['name']))


def fix_coco_datasetEntry():
    parser = argparse.ArgumentParser()
    parser.add_argument('ann',  help='input paths')
    args = parser.parse_args()

    # 读取COCO标注文件
    coco = COCO(args.ann)

    # 获取指定类别的ID
    category_ids = coco.getCatIds()

    # 获取所有图像ID
    all_img_ids = coco.getImgIds()

    # 初始化保存合法的图像ID和标注的列表
    valid_img_ids = []
    valid_annotations = []

    for img_id in all_img_ids:
        ann_ids = coco.getAnnIds(
            imgIds=[img_id], catIds=category_ids, iscrowd=None)
        anns = coco.loadAnns(ann_ids)

        # 如果图像有合法的标注，记录该图像ID和标注
        tmp_valid_anns = []
        for ann in anns:
            if "bbox" in ann.keys() and len(ann["bbox"]) == 4 and (ann['bbox'][2] * ann['bbox'][3]) >= 10 * 10:
                tmp_valid_anns.append(ann)

        if tmp_valid_anns:
            valid_img_ids.append(img_id)
            valid_annotations.extend(tmp_valid_anns)

    # 获取所有信息的字典
    coco_info = coco.dataset.get('info', {})
    coco_licenses = coco.dataset.get('licenses', [])

    # 筛选合法图像
    valid_images = coco.loadImgs(valid_img_ids)

    # 筛选指定类别的类别信息
    valid_categories = coco.loadCats(category_ids)

    # 创建新的COCO格式的字典
    filtered_coco = {
        'info': coco_info,
        'licenses': coco_licenses,
        'images': valid_images,
        'annotations': valid_annotations,
        'categories': valid_categories
    }

    # 将结果写入一个新的JSON文件
    with open(args.ann, 'w') as f:
        json.dump(filtered_coco, f)


def change_img_dir_related_to(annfile, base_dir):
    print("rebase to ", base_dir)
    coco = COCO(annfile)

    for img_id in coco.getImgIds():
        img = coco.loadImgs(img_id)[0]
        img_path = img['file_name']

        # Modify image path to relative path
        img['file_name'] = os.path.relpath(img_path, base_dir)

        # Update image annotation
        coco.loadImgs(img_id)[0] = img

# Save modified annotation file
    new_annFile = annfile
    with open(new_annFile, 'w') as f:
        json.dump(coco.dataset, f)


def mergeEntry():
    parser = argparse.ArgumentParser()
    parser.add_argument('ann0',  help='input paths')
    parser.add_argument('img0',  help='input paths')
    parser.add_argument('ann1',  help='input paths')
    parser.add_argument('img1',  help='input paths')
    parser.add_argument('merge_out', help='merge output paths')
    args = parser.parse_args()

# init Coco objects by specifying coco dataset paths and image folder directories
    coco_1 = Coco.from_coco_dict_or_path(args.ann0, image_dir=args.img0)
    coco_2 = Coco.from_coco_dict_or_path(args.ann1, image_dir=args.img1)

# merge Coco datasets
    coco_1.merge(coco_2)
    save_json(coco_1.json, args.merge_out)
    change_img_dir_related_to(args.merge_out, os.path.dirname(args.merge_out))
    plotstatisc(args.merge_out, False)


def sliceEntry():
    from sahi.slicing import slice_coco
    parser = argparse.ArgumentParser()
    parser.add_argument('img_dir',  help='image dir')
    parser.add_argument('ann_path',  help='input paths')
    parser.add_argument('height',  help='slice height')
    parser.add_argument('width',  help='slice widt')
    parser.add_argument('overlap', default=0.2,  help='slice overlap')
    args = parser.parse_args()

    coco_dict, coco_path = slice_coco(
        coco_annotation_file_path=args.ann_path,
        image_dir=args.img_dir,
        slice_height=args.height,
        slice_width=args.width,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )


def cocoemtpyEntry():
    from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
    from sahi.utils.file import save_json
    parser = argparse.ArgumentParser()
    parser.add_argument('ouput_json',  help='output json  path')
    args = parser.parse_args()

    coco = Coco()
    coco.add_category(CocoCategory(id=1, name='car'))
    save_json(coco.json, args.ouput_json)


def splitEntry():
    from sahi.utils.coco import Coco
    from sahi.utils.file import save_json
    parser = argparse.ArgumentParser()
    parser.add_argument('ann_json',  help='ann json  path')
    parser.add_argument('--train_ratio', default=0.85, help='ann json  path')
    args = parser.parse_args()

# specify coco dataset path

# init Coco object
    coco = Coco.from_coco_dict_or_path(args.ann_json)

# split COCO dataset with a 85% train/15% val split
    result = coco.split_coco_as_train_val(
        train_split_rate=args.train_ratio
    )

    train_coco_json_path = args.ann_json.replace(".json", "_train.json")
    val_coco_json_path = args.ann_json.replace(".json", "_val.json")
# export train val split files
    save_json(result["train_coco"].json, train_coco_json_path)
    save_json(result["val_coco"].json, val_coco_json_path)


def extractCatEntry():
    from sahi.utils.coco import Coco
    from sahi.utils.file import save_json
    parser = argparse.ArgumentParser()
    parser.add_argument('ann_json',  help='ann json path')
    parser.add_argument('out_json',  help='out json path')
    parser.add_argument('cat_names', nargs='*', type=str,
                        help='desired categories')
    args = parser.parse_args()

# specify coco dataset path
    print(args.cat_names)
    desired_name2id = {n: i + 1 for i, n in enumerate(args.cat_names)}

# init Coco object
    coco = Coco.from_coco_dict_or_path(args.ann_json)

    ori_mapping = coco.category_mapping
    pprint.pprint("ori mapping is ")
    pprint.pprint(ori_mapping)

    pprint.pprint("new mapping is ")
    pprint.pprint(desired_name2id)

# split COCO dataset with a 85% train/15% val split
    coco.update_categories(desired_name2id)

    save_json(coco.json, args.out_json)


def results2AnnEntry():
    parser = argparse.ArgumentParser()
    parser.add_argument('ann_json',  help='ann json path')
    parser.add_argument('res_json',  help='res json path')
    parser.add_argument('out_json',  help='out json path')
    args = parser.parse_args()

    coco_gt = COCO(args.ann_json)
    coco_results = coco_gt.loadRes(args.res_json)
    coco_json_results = []
    for result in coco_results.dataset['annotations']:
        coco_json_results.append({
            'image_id': result['image_id'],
            'category_id': result['category_id'],
            'bbox': result['bbox'],
            'score': result['score'],
        })
    coco_json_formatted_results = {
        'info': coco_results.dataset.get('info', {}),
        'images': coco_results.dataset.get('images', []),
        'categories': coco_results.dataset['categories'],
        'annotations': coco_json_results
    }

    with open(args.out_json, 'w') as f:
        json.dump(coco_json_formatted_results, f)


def cocoSmapleEntry():
    parser = argparse.ArgumentParser()
    parser.add_argument("ann_path")
    parser.add_argument("img_dir")
    parser.add_argument("num")
    parser.add_argument("new_img_dir")
    parser.add_argument("new_ann_path")
    args = parser.parse_args()


# specify coco dataset path
    coco_path = args.ann_path
    img_dir = args.img_dir

# init Coco object
    coco = Coco.from_coco_dict_or_path(coco_path)
    img_nums = len(coco.images)
# create a Coco object with 1/10 of total images
    dst_nums = int(args.num)
    subsampled_coco = coco.get_subsampled_coco(
        subsample_ratio=int(img_nums/dst_nums))


# process dir
    if not os.path.isdir(args.new_img_dir):
        os.system("mkdir -p {}".format(args.new_img_dir))

    jimages = subsampled_coco.json["images"]

    for jj in tqdm(jimages):
        ori_img_path = os.path.join(img_dir, jj["file_name"])
        base_name = os.path.basename(ori_img_path)
        jj["file_name"] = base_name

        dst_img_path = os.path.join(args.new_img_dir, jj["file_name"])
        os.system(
            "cp {} {}".format(ori_img_path, dst_img_path))

    save_json(subsampled_coco.json, args.new_ann_path)
