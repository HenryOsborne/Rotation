import numpy as np
import torch
import torch.nn as nn


def re_order_torch(bboxes, with_label=False):
    n = len(bboxes)
    targets = []
    for i in range(n):
        box = bboxes[i]
        # 寻找x1
        x1 = box[0]
        y1 = box[1]
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
    targets = torch.stack([torch.stack(i) for i in targets])
    return targets


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, classifications, regressions, anchors, annotations_h, annotations_r):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        # anchor_widths = anchor[:, 2] - anchor[:, 0]
        # anchor_heights = anchor[:, 3] - anchor[:, 1]
        # anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_widths
        # anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_heights
        anchor_ctr_x = (anchor[:, 2] + anchor[:, 0]) / 2
        anchor_ctr_y = (anchor[:, 3] + anchor[:, 1]) / 2
        anchor_widths = anchor[:, 2] - anchor[:, 0] + 1
        anchor_heights = anchor[:, 3] - anchor[:, 1] + 1

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annot_h = annotations_h[j, :, :]
            bbox_annot_h = bbox_annot_h[bbox_annot_h[:, 4] != -1]

            bbox_annot_r = annotations_r[j, :, :]
            bbox_annot_r = bbox_annot_r[bbox_annot_r[:, 8] != -1]

            classification = torch.clamp(classification, 1e-4, 1.0 - 1e-4)

            assert bbox_annot_h.shape[0] == bbox_annot_r.shape[0]
            if bbox_annot_h.shape[0] == 0:
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float().cuda())

                else:
                    alpha_factor = torch.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

                    bce = -(torch.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(torch.tensor(0).float())

                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annot_h[:, :4])  # num_anchors x num_annotations

            IoU_max, IoU_argmax = torch.max(IoU, dim=1)  # num_anchors x 1

            # compute the loss for classification
            anchor_state = torch.ones(classification.shape) * -1

            if torch.cuda.is_available():
                anchor_state = anchor_state.cuda()

            # IoU小于0.4设置为0
            anchor_state[torch.lt(IoU_max, 0.4), :] = 0

            # IoU大于0.5，设置为正样本
            positive_indices = torch.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations_h = bbox_annot_h[IoU_argmax, :]
            assigned_annotations_r = bbox_annot_r[IoU_argmax, :]

            anchor_state[positive_indices, :] = 0
            anchor_state[positive_indices, assigned_annotations_r[positive_indices, 8].long()] = 1

            if torch.cuda.is_available():
                alpha_factor = torch.ones(anchor_state.shape).cuda() * alpha
            else:
                alpha_factor = torch.ones(anchor_state.shape) * alpha

            alpha_factor = torch.where(torch.eq(anchor_state, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(anchor_state, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * torch.pow(focal_weight, gamma)

            bce = -(anchor_state * torch.log(classification) + (1.0 - anchor_state) * torch.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(anchor_state, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(anchor_state, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum() / torch.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression
            sigma = 3.0
            sigma_squared = sigma ** 2

            if positive_indices.sum() > 0:
                preds = regression[positive_indices, :]
                targets = assigned_annotations_r[positive_indices, :]
                anchor_widths_pi = anchor_widths[positive_indices]
                anchor_heights_pi = anchor_heights[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                dx1 = preds[:, 0]
                dy1 = preds[:, 1]
                dx2 = preds[:, 2]
                dy2 = preds[:, 3]
                dx3 = preds[:, 4]
                dy3 = preds[:, 5]
                dx4 = preds[:, 6]
                dy4 = preds[:, 7]

                pred_x_1 = dx1 * anchor_widths_pi + anchor_ctr_x_pi
                pred_y_1 = dy1 * anchor_heights_pi + anchor_ctr_y_pi
                pred_x_2 = dx2 * anchor_widths_pi + anchor_ctr_x_pi
                pred_y_2 = dy2 * anchor_heights_pi + anchor_ctr_y_pi
                pred_x_3 = dx3 * anchor_widths_pi + anchor_ctr_x_pi
                pred_y_3 = dy3 * anchor_heights_pi + anchor_ctr_y_pi
                pred_x_4 = dx4 * anchor_widths_pi + anchor_ctr_x_pi
                pred_y_4 = dy4 * anchor_heights_pi + anchor_ctr_y_pi

                preds = torch.stack([pred_x_1, pred_y_1,
                                     pred_x_2, pred_y_2,
                                     pred_x_3, pred_y_3,
                                     pred_x_4, pred_y_4],
                                    dim=1)

                targets = re_order_torch(targets, with_label=True)

                normalizer = max(1.0, float(num_positive_anchors))

                # loss1
                loss1_1 = (preds[:, 0] - targets[:, 0]) / anchor_widths_pi
                loss1_2 = (preds[:, 1] - targets[:, 1]) / anchor_heights_pi
                loss1_3 = (preds[:, 2] - targets[:, 2]) / anchor_widths_pi
                loss1_4 = (preds[:, 3] - targets[:, 3]) / anchor_heights_pi
                loss1_5 = (preds[:, 4] - targets[:, 4]) / anchor_widths_pi
                loss1_6 = (preds[:, 5] - targets[:, 5]) / anchor_heights_pi
                loss1_7 = (preds[:, 6] - targets[:, 6]) / anchor_widths_pi
                loss1_8 = (preds[:, 7] - targets[:, 7]) / anchor_heights_pi
                box_diff_1 = torch.stack([loss1_1, loss1_2, loss1_3, loss1_4, loss1_5, loss1_6, loss1_7, loss1_8],
                                         dim=1)
                box_diff_1 = torch.abs(box_diff_1)
                loss_1 = torch.where(box_diff_1 < (1.0 / sigma_squared), 0.5 * sigma_squared * torch.pow(box_diff_1, 2),
                                     box_diff_1 - 0.5 / sigma_squared)
                loss_1 = loss_1.sum(dim=1)

                # loss2
                loss2_1 = (preds[:, 0] - targets[:, 2]) / anchor_widths_pi
                loss2_2 = (preds[:, 1] - targets[:, 3]) / anchor_heights_pi
                loss2_3 = (preds[:, 2] - targets[:, 4]) / anchor_widths_pi
                loss2_4 = (preds[:, 3] - targets[:, 5]) / anchor_heights_pi
                loss2_5 = (preds[:, 4] - targets[:, 6]) / anchor_widths_pi
                loss2_6 = (preds[:, 5] - targets[:, 7]) / anchor_heights_pi
                loss2_7 = (preds[:, 6] - targets[:, 0]) / anchor_widths_pi
                loss2_8 = (preds[:, 7] - targets[:, 1]) / anchor_heights_pi
                box_diff_2 = torch.stack([loss2_1, loss2_2, loss2_3, loss2_4, loss2_5, loss2_6, loss2_7, loss2_8], 1)
                box_diff_2 = torch.abs(box_diff_2)
                loss_2 = torch.where(box_diff_2 < 1.0 / sigma_squared, 0.5 * sigma_squared * torch.pow(box_diff_2, 2),
                                     box_diff_2 - 0.5 / sigma_squared)
                loss_2 = loss_2.sum(dim=1)

                # loss3
                loss3_1 = (preds[:, 0] - targets[:, 6]) / anchor_widths_pi
                loss3_2 = (preds[:, 1] - targets[:, 7]) / anchor_heights_pi
                loss3_3 = (preds[:, 2] - targets[:, 0]) / anchor_widths_pi
                loss3_4 = (preds[:, 3] - targets[:, 1]) / anchor_heights_pi
                loss3_5 = (preds[:, 4] - targets[:, 2]) / anchor_widths_pi
                loss3_6 = (preds[:, 5] - targets[:, 3]) / anchor_heights_pi
                loss3_7 = (preds[:, 6] - targets[:, 4]) / anchor_widths_pi
                loss3_8 = (preds[:, 7] - targets[:, 5]) / anchor_heights_pi
                box_diff_3 = torch.stack([loss3_1, loss3_2, loss3_3, loss3_4, loss3_5, loss3_6, loss3_7, loss3_8],
                                         dim=1)
                box_diff_3 = torch.abs(box_diff_3)
                loss_3 = torch.where(box_diff_3 < 1.0 / sigma_squared, 0.5 * sigma_squared * torch.pow(box_diff_3, 2),
                                     box_diff_3 - 0.5 / sigma_squared)
                loss_3 = loss_3.sum(dim=1)

                regression_loss = torch.min(torch.min(loss_1, loss_2), loss_3)
                regression_loss = torch.sum(regression_loss) / normalizer
                regression_losses.append(regression_loss.mean())

        return torch.stack(classification_losses).mean(dim=0, keepdim=True), \
               torch.stack(regression_losses).mean(dim=0, keepdim=True)


'''
assigned_annotations_h = assigned_annotations_h[positive_indices, :]

anchor_widths_pi = anchor_widths[positive_indices]
anchor_heights_pi = anchor_heights[positive_indices]
anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

gt_widths = assigned_annotations_h[:, 2] - assigned_annotations_h[:, 0]
gt_heights = assigned_annotations_h[:, 3] - assigned_annotations_h[:, 1]
gt_ctr_x = assigned_annotations_h[:, 0] + 0.5 * gt_widths
gt_ctr_y = assigned_annotations_h[:, 1] + 0.5 * gt_heights

# clip widths to 1
gt_widths = torch.clamp(gt_widths, min=1)
gt_heights = torch.clamp(gt_heights, min=1)

targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
targets_dw = torch.log(gt_widths / anchor_widths_pi)
targets_dh = torch.log(gt_heights / anchor_heights_pi)

anchor_state = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh))
anchor_state = anchor_state.t()

if torch.cuda.is_available():
    anchor_state = anchor_state / torch.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
else:
    anchor_state = anchor_state / torch.Tensor([[0.1, 0.1, 0.2, 0.2]])

negative_indices = 1 + (~positive_indices)

regression_diff = torch.abs(anchor_state - regression[positive_indices, :])

regression_loss = torch.where(
    torch.le(regression_diff, 1.0 / 9.0),
    0.5 * 9.0 * torch.pow(regression_diff, 2),
    regression_diff - 0.5 / 9.0
)
regression_losses.append(regression_loss.mean())

else:
if torch.cuda.is_available():
regression_losses.append(torch.tensor(0).float().cuda())
else:
regression_losses.append(torch.tensor(0).float())
'''
