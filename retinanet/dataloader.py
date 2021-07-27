from __future__ import print_function, division
import sys
import os

import cv2
import torch
import numpy as np
import random
import csv

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

from pycocotools.coco import COCO

import skimage.io
import skimage.transform
import skimage.color
import skimage
from skimage import img_as_ubyte, img_as_float
from skimage import img_as_float

from PIL import Image
from .utils import re_order


def skimage2opencv(src):
    cv_image = img_as_ubyte(src)
    return cv_image


def opencv2skimage(src):
    image = img_as_float(src)
    return image


class DotaDataset(Dataset):
    """Dota dataset."""

    def __init__(self, root_dir, set_name='images', transform=None, show=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.show = show
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'DOTA_trainval600.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot_h, annot_r = self.load_annotations(idx)
        if self.show:
            self.show_box(img, annot_h, annot_r)
        sample = {'img': img, 'annot_h': annot_h, 'annot_r': annot_r}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images', image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        gt_and_label_h = np.zeros((0, 5))
        gt_and_label_r = np.zeros((0, 9))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return gt_and_label_h, gt_and_label_r

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            h_annotation = np.zeros((1, 5))
            h_annotation[0, :4] = a['bbox']
            h_annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            gt_and_label_h = np.append(gt_and_label_h, h_annotation, axis=0)

            r_annotation = np.zeros((1, 9))
            r_annotation[0, :8] = a['segmentation'][0]
            r_annotation[0, 8] = self.coco_label_to_label(a['category_id'])
            gt_and_label_r = np.append(gt_and_label_r, r_annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        gt_and_label_h[:, 2] = gt_and_label_h[:, 0] + gt_and_label_h[:, 2]
        gt_and_label_h[:, 3] = gt_and_label_h[:, 1] + gt_and_label_h[:, 3]

        return gt_and_label_h, gt_and_label_r

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 15

    def show_box(self, img, annot_h, annot_r):
        img = skimage2opencv(img)
        for i in range(len(annot_h)):
            a_h = list(map(int, annot_h[i]))
            a_r = list(map(int, annot_r[i]))
            cv2.rectangle(img, (a_h[0], a_h[1]), (a_h[2], a_h[3]), (255, 0, 0))
            cv2.line(img, (a_r[0], a_r[1]), (a_r[2], a_r[3]), (0, 255, 0))
            cv2.circle(img, (a_r[0], a_r[1]), radius=5, color=(0, 255, 0))
            cv2.line(img, (a_r[2], a_r[3]), (a_r[4], a_r[5]), (0, 255, 0))
            cv2.line(img, (a_r[4], a_r[5]), (a_r[6], a_r[7]), (0, 255, 0))
            cv2.line(img, (a_r[6], a_r[7]), (a_r[0], a_r[1]), (0, 255, 0))
        cv2.imshow('demo', img)
        cv2.waitKey()
        cv2.destroyAllWindows()


class CocoDataset(Dataset):
    """Coco dataset."""

    def __init__(self, root_dir, set_name='train2017', transform=None):
        """
        Args:
            root_dir (string): COCO directory.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        self.coco_labels = {}
        self.coco_labels_inverse = {}
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, 'images', self.set_name, image_info['file_name'])
        img = skimage.io.imread(path)

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    def num_classes(self):
        return 80


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, class_list, transform=None):
        self.train_file = train_file
        self.class_list = class_list
        self.transform = transform

        # parse the provided class file
        try:
            with self._open_for_csv(self.class_list) as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise (ValueError('invalid CSV class file: {}: {}'.format(self.class_list, e)))

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # csv with img_path, x1, y1, x2, y2, class_name
        try:
            with self._open_for_csv(self.train_file) as file:
                self.image_data = self._read_annotations(csv.reader(file, delimiter=','), self.classes)
        except ValueError as e:
            raise (ValueError('invalid CSV annotations file: {}: {}'.format(self.train_file, e)))
        self.image_names = list(self.image_data.keys())

    def _parse(self, value, function, fmt):
        """
        Parse a string into a value, and format a nice ValueError if it fails.
        Returns `function(value)`.
        Any `ValueError` raised is catched and a new `ValueError` is raised
        with message `fmt.format(e)`, where `e` is the caught `ValueError`.
        """
        try:
            return function(value)
        except ValueError as e:
            raise (ValueError(fmt.format(e)), None)

    def _open_for_csv(self, path):
        """
        Open a file with flags suitable for csv.reader.
        This is different for python2 it means with mode 'rb',
        for python3 this means 'r' with "universal newlines".
        """
        if sys.version_info[0] < 3:
            return open(path, 'rb')
        else:
            return open(path, 'r', newline='')

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row
            except ValueError:
                raise (ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
            class_id = self._parse(class_id, int, 'line {}: malformed class ID: {{}}'.format(line))

            if class_name in result:
                raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        img = skimage.io.imread(self.image_names[image_index])

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_names[image_index]]
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations (like image with id 257034)
        if len(annotation_list) == 0:
            return annotations

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1']
            x2 = a['x2']
            y1 = a['y1']
            y2 = a['y2']

            if (x2 - x1) < 1 or (y2 - y1) < 1:
                continue

            annotation = np.zeros((1, 5))

            annotation[0, 0] = x1
            annotation[0, 1] = y1
            annotation[0, 2] = x2
            annotation[0, 3] = y2

            annotation[0, 4] = self.name_to_label(a['class'])
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def _read_annotations(self, csv_reader, classes):
        result = {}
        for line, row in enumerate(csv_reader):
            line += 1

            try:
                img_file, x1, y1, x2, y2, class_name = row[:6]
            except ValueError:
                raise (ValueError(
                    'line {}: format should be \'img_file,x1,y1,x2,y2,class_name\' or \'img_file,,,,,\''.format(line)),
                       None)

            if img_file not in result:
                result[img_file] = []

            # If a row contains only an image path, it's an image without annotations.
            if (x1, y1, x2, y2, class_name) == ('', '', '', '', ''):
                continue

            x1 = self._parse(x1, int, 'line {}: malformed x1: {{}}'.format(line))
            y1 = self._parse(y1, int, 'line {}: malformed y1: {{}}'.format(line))
            x2 = self._parse(x2, int, 'line {}: malformed x2: {{}}'.format(line))
            y2 = self._parse(y2, int, 'line {}: malformed y2: {{}}'.format(line))

            # Check that the bounding box is valid.
            if x2 <= x1:
                raise ValueError('line {}: x2 ({}) must be higher than x1 ({})'.format(line, x2, x1))
            if y2 <= y1:
                raise ValueError('line {}: y2 ({}) must be higher than y1 ({})'.format(line, y2, y1))

            # check if the current class name is correctly present
            if class_name not in classes:
                raise ValueError('line {}: unknown class name: \'{}\' (classes: {})'.format(line, class_name, classes))

            result[img_file].append({'x1': x1, 'x2': x2, 'y1': y1, 'y2': y2, 'class': class_name})
        return result

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(self.image_names[image_index])
        return float(image.width) / float(image.height)


def show_(img, annot_h, annot_r):
    transform = transforms.Compose([UnNormalizer()])
    new_img = transform(img)
    new_img = new_img.permute(1, 2, 0)
    new_img = new_img.numpy()

    annot_h = annot_h.numpy()
    annot_r = annot_r.numpy()
    new_img = skimage2opencv(new_img)

    for i in range(len(annot_h)):
        a_h = list(map(int, annot_h[i]))
        a_r = list(map(int, annot_r[i]))
        cv2.rectangle(new_img, (a_h[0], a_h[1]), (a_h[2], a_h[3]), (255, 0, 0))
        cv2.line(new_img, (a_r[0], a_r[1]), (a_r[2], a_r[3]), (0, 255, 0))
        cv2.circle(new_img, (a_r[0], a_r[1]), radius=5, color=(0, 255, 0))
        cv2.line(new_img, (a_r[2], a_r[3]), (a_r[4], a_r[5]), (0, 255, 0))
        cv2.circle(new_img, (a_r[2], a_r[3]), radius=5, color=(0, 0, 255))
        cv2.line(new_img, (a_r[4], a_r[5]), (a_r[6], a_r[7]), (0, 255, 0))
        cv2.line(new_img, (a_r[6], a_r[7]), (a_r[0], a_r[1]), (0, 255, 0))
    cv2.imshow('demo', new_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def collater(data):
    imgs = [s['img'] for s in data]
    annots_h = [s['annot_h'] for s in data]
    annots_r = [s['annot_r'] for s in data]
    # TODO:将旋转框按逆时针排序，但是起始点不是左上角
    annots_r = [re_order(annot_r, with_label=True) for annot_r in annots_r]
    annots_h = [torch.from_numpy(ann) for ann in annots_h]
    annots_r = [torch.from_numpy(ann) for ann in annots_r]
    scales = [s['scale'] for s in data]

    assert len(annots_h) == len(annots_r)

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots_h)

    if max_num_annots > 0:
        annot_padded_h = torch.ones((len(annots_h), max_num_annots, 5)) * -1
        annot_padded_r = torch.ones((len(annots_h), max_num_annots, 9)) * -1
        if max_num_annots > 0:
            for idx, (annot_h, annot_r) in enumerate(zip(annots_h, annots_r)):
                # print(annot_h.shape)
                if annot_h.shape[0] > 0:
                    annot_padded_h[idx, :annot_h.shape[0], :] = annot_h
                if annot_r.shape[0] > 0:
                    annot_padded_r[idx, :annot_r.shape[0], :] = annot_r
    else:
        annot_padded_h = torch.ones((len(annots_h), 1, 5)) * -1
        annot_padded_r = torch.ones((len(annots_h), 1, 9)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    if len(scales) == 1:  # 在batch size为1时显示图片和GT BOX
        show_(padded_imgs.squeeze(0), annot_padded_h.squeeze(0), annot_padded_r.squeeze(0))

    return {'img': padded_imgs, 'annot_h': annot_padded_h, 'annot_r': annot_padded_r, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots_h, annots_r = sample['img'], sample['annot_h'], sample['annot_r']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots_h[:, :4] *= scale
        annots_r[:, :8] *= scale

        annots_r = re_order(annots_r, with_label=True)

        return {'img': torch.from_numpy(new_image),
                'annot_h': annots_h,
                'annot_r': annots_r,
                'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annot_h, annot_r = sample['img'], sample['annot_h'], sample['annot_r']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annot_h[:, 0].copy()
            x2 = annot_h[:, 2].copy()

            x_tmp = x1.copy()

            annot_h[:, 0] = cols - x2
            annot_h[:, 2] = cols - x_tmp

            x1, y1, x2, y2, x3, y3, x4, y4, label = np.split(annot_r, 9, axis=1)
            new_x1 = cols - x1
            new_x2 = cols - x2
            new_x3 = cols - x3
            new_x4 = cols - x4
            annot_r = np.concatenate((new_x1, y1, new_x2, y2, new_x3, y3, new_x4, y4, label), axis=1)

            sample = {'img': image, 'annot_h': annot_h, 'annot_r': annot_r}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annot_h, annot_r = sample['img'], sample['annot_h'], sample['annot_r']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot_h': annot_h, 'annot_r': annot_r}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean is None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std is None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]


if __name__ == '__main__':
    ############################### 在随机水平反转之后re order旋转框 ###############################
    dataset_train = DotaDataset('data/dota_tiny', set_name='images',
                                transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]),
                                show=False)
    sampler = AspectRatioBasedSampler(dataset_train, batch_size=1, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=0, collate_fn=collater, batch_sampler=sampler)
    for iter_num, data in enumerate(dataloader_train):
        print(data['annot_h'])
        print(data['annot_r'])