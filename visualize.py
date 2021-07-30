import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import sys
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer, DotaDataset

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.', default='dota')
    parser.add_argument('--coco_path', help='Path to COCO directory', default='data/dota_tiny')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--model', help='Path to model (.pt) file.', default='work_dir/dota_retinanet_49.pt')

    parser = parser.parse_args(args)

    if parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.coco_path, set_name='train2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'csv':
        dataset_val = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'dota':
        dataset_val = DotaDataset(parser.coco_path, set_name='images',
                                  transform=transforms.Compose([Normalizer(), Resizer()]), show=False)
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    retinanet = torch.load(parser.model)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.eval()

    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                batch_scores, batch_labals, batch_bbox_pred = retinanet(data['img'].cuda().float())
            else:
                batch_scores, batch_labals, batch_bbox_pred = retinanet(data['img'].float())
            print('Elapsed time: {}'.format(time.time() - st))

            for i in range(len(batch_scores)):
                scores, label, bbox_pred = batch_scores[i], batch_labals[i], batch_bbox_pred[i]

                idxs = np.where(scores.cpu() > 0.5)[0]

                img = np.array(255 * unnormalize(data['img'][i, :, :, :])).copy()
                img[img < 0] = 0
                img[img > 255] = 255
                img = np.transpose(img, (1, 2, 0))
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

                for j in range(idxs.shape[0]):
                    bbox = bbox_pred[idxs[j], :]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    x3 = int(bbox[4])
                    y3 = int(bbox[5])
                    x4 = int(bbox[6])
                    y4 = int(bbox[7])
                    label_name = dataset_val.labels[int(label[idxs[j]]) - 1]
                    draw_caption(img, (x1, y1, x2, y2), label_name)

                    cv2.line(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                    cv2.line(img, (x2, y2), (x3, y3), color=(0, 0, 255), thickness=2)
                    cv2.line(img, (x3, y3), (x4, y4), color=(0, 0, 255), thickness=2)
                    cv2.line(img, (x4, y4), (x1, y1), color=(0, 0, 255), thickness=2)

                cv2.imshow('img', img)
                cv2.waitKey()
                cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
