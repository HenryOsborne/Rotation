import torch
import torch.nn as nn
import numpy as np


def find_x1(box):
    # TODO:need to debug
    ymin_idx = np.argmin(box[1::2])
    y1 = box[ymin_idx * 2 + 1]
    x1 = box[ymin_idx * 2]
    np.hstack((box[ymin_idx * 2:], box[:ymin_idx * 2]))  # reorder box
    return x1, y1, box


# counterclockwise, write by WenQian
def re_order(bboxes, with_label=False):
    n = len(bboxes)
    targets = []
    for i in range(n):
        box = bboxes[i]
        # 寻找x1
        x1 = box[0]
        y1 = box[1]
        # x1, y1, box = find_x1(box)
        x1_index = 0
        for j in range(1, 4):
            ### if x larger than x1 then continue
            if box[2 * j] > x1:
                continue
            ### if x smaller than x1 then replace x1 as x
            elif box[2 * j] < x1:
                x1 = box[2 * j]
                y1 = box[2 * j + 1]
                x1_index = j
            ### if they are euqal then we aims to find the upper point
            else:
                if box[2 * j + 1] < y1:
                    x1 = box[2 * j]
                    y1 = box[2 * j + 1]
                    x1_index = j
                else:
                    continue

        # 寻找与x1连线中间点
        for j in range(4):
            if j == x1_index:
                continue
            x_ = box[2 * j]
            y_ = box[2 * j + 1]
            x_index = j
            val = []
            for k in range(4):
                if k == x_index or k == x1_index:
                    continue
                else:
                    x = box[2 * k]
                    y = box[2 * k + 1]
                    if x1 == x_:
                        val.append(x - x1)
                    else:
                        val1 = (y - y1) - (y_ - y1) / (x_ - x1) * (x - x1)
                        val.append(val1)
            if val[0] * val[1] < 0:
                x3 = x_
                y3 = y_
                for k in range(4):
                    if k == x_index or k == x1_index:
                        continue
                    x = box[2 * k]
                    y = box[2 * k + 1]
                    if not x1 == x3:
                        val = (y - y1) - (y3 - y1) / (x3 - x1) * (x - x1)
                        if val >= 0:
                            x2 = x
                            y2 = y
                        if val < 0:
                            x4 = x
                            y4 = y
                    else:
                        val = x1 - x
                        if val >= 0:
                            x2 = x
                            y2 = y
                        if val < 0:
                            x4 = x
                            y4 = y
                break
        try:
            if with_label:
                targets.append([x1, y1, x2, y2, x3, y3, x4, y4, box[-1]])
            else:
                targets.append([x1, y1, x2, y2, x3, y3, x4, y4])
        except:
            print('**' * 20)
            print(box)
            targets.append(box)
    return np.array(targets, np.float32)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BBoxTransform(nn.Module):

    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()
        if mean is None:
            if torch.cuda.is_available():
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32)).cuda()
            else:
                self.mean = torch.from_numpy(np.array([0, 0, 0, 0]).astype(np.float32))

        else:
            self.mean = mean
        if std is None:
            if torch.cuda.is_available():
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)).cuda()
            else:
                self.std = torch.from_numpy(np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))
        else:
            self.std = std

    def forward(self, boxes, deltas):

        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]
        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = torch.exp(dw) * widths
        pred_h = torch.exp(dh) * heights

        pred_boxes_x1 = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y1 = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x2 = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y2 = pred_ctr_y + 0.5 * pred_h

        pred_boxes = torch.stack([pred_boxes_x1, pred_boxes_y1, pred_boxes_x2, pred_boxes_y2], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):

    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = torch.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = torch.clamp(boxes[:, :, 1], min=0)

        boxes[:, :, 2] = torch.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = torch.clamp(boxes[:, :, 3], max=height)

        return boxes
