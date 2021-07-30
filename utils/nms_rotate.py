# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import sys

import torch

sys.path.append('../../')
from utils.rotate_polygon_nms import rotate_gpu_nms


def nms_rotate(decode_boxes, scores, iou_threshold, max_output_size, use_gpu=True, gpu_id=0):
    """
    :param boxes: format [x_c, y_c, w, h, theta]
    :param scores: scores of boxes
    :param threshold: iou threshold (0.7 or 0.5)
    :param max_output_size: max number of output
    :return: the remaining index of boxes
    """

    if use_gpu:
        keep = nms_rotate_gpu(boxes_list=decode_boxes,
                              scores=scores,
                              iou_threshold=iou_threshold,
                              device_id=gpu_id,
                              max_output_size=max_output_size)

        if keep.shape[0] > max_output_size:
            keep = keep[max_output_size]
        else:
            keep = keep
        # 如果keep的数量大于max_output_size，取前max_output_size个，否则保持keep不变
    else:
        keep = nms_rotate_cpu(decode_boxes, scores, iou_threshold, max_output_size)
    return keep


def nms_rotate_cpu(boxes, scores, iou_threshold, max_output_size):
    keep = []

    order = scores.argsort()[::-1]
    num = boxes.shape[0]

    suppressed = np.zeros((num), dtype=int)

    for _i in range(num):
        if len(keep) >= max_output_size:
            break

        i = order[_i]
        if suppressed[i] == 1:
            continue
        keep.append(i)
        r1 = ((boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3]), boxes[i, 4])
        area_r1 = boxes[i, 2] * boxes[i, 3]
        for _j in range(_i + 1, num):
            j = order[_j]
            if suppressed[i] == 1:
                continue

            if np.sqrt((boxes[i, 0] - boxes[j, 0]) ** 2 + (boxes[i, 1] - boxes[j, 1]) ** 2) > (
                    boxes[i, 2] + boxes[j, 2] + boxes[i, 3] + boxes[j, 3]):
                inter = 0.0
            else:

                r2 = ((boxes[j, 0], boxes[j, 1]), (boxes[j, 2], boxes[j, 3]), boxes[j, 4])
                area_r2 = boxes[j, 2] * boxes[j, 3]
                inter = 0.0

                try:
                    int_pts = cv2.rotatedRectangleIntersection(r1, r2)[1]

                    if int_pts is not None:
                        order_pts = cv2.convexHull(int_pts, returnPoints=True)

                        int_area = cv2.contourArea(order_pts)

                        inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + 1e-4)

                except:
                    """
                      cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
                      error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
                    """
                    # print(r1)
                    # print(r2)
                    inter = 0.9999

            if inter >= iou_threshold:
                suppressed[j] = 1

    return np.array(keep, np.int64)


def rnms_gpu(det_boxes, iou_threshold, device_id):
    if det_boxes.shape[0] == 0:
        return np.array([], np.int64)
    else:
        assert det_boxes.shape[1] == 6, 'shape of det_boxes is not 6, {}'.format(det_boxes)
        keep = rotate_gpu_nms(det_boxes, iou_threshold, device_id)
        keep = np.reshape(keep, [-1])
        return np.array(keep, np.int64)


def nms_rotate_gpu(boxes_list, scores, iou_threshold, device_id=0, max_output_size=100):
    if boxes_list.shape[0] == 0:
        return np.array([], np.int64)
    else:
        det_tensor = np.concatenate([boxes_list, np.expand_dims(scores, axis=1)], axis=1)
        keep = rnms_gpu(det_tensor, iou_threshold, device_id)
        return keep


if __name__ == '__main__':
    boxes = np.array([[50, 50, 100, 100, 0],
                      [60, 60, 100, 100, 0],
                      [50, 50, 100, 100, -45.],
                      [200, 200, 100, 100, 0.]])
    boxes = np.tile(boxes, (10000, 1)).astype(np.float32)

    scores = np.array([0.99, 0.88, 0.66, 0.77]).astype(np.float32)
    scores = np.tile(scores, (10000))
    print(boxes.shape)

    keep = nms_rotate(boxes, scores, iou_threshold=0.7, max_output_size=5, use_gpu=False)

    print(keep)
