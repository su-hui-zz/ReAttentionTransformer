# -----------------------------------------------------------------------------------------------------
# Re-Attention Transformer using maxboxaccv2 is implemented based on https://github.com/clovaai/wsolevaluation.git
# -----------------------------------------------------------------------------------------------------

"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import cv2
import numpy as np
import os
import torch
from os.path import join as ospj
import torch.utils.data as torchdata
import pdb

_IMAGENET_MEAN = [0.485, .456, .406]
_IMAGENET_STDDEV = [.229, .224, .225]
_RESIZE_LENGTH = 224
_CONTOUR_INDEX = 1 if cv2.__version__.split('.')[0] == '3' else 0


def resize_cam(cam, size=(224, 224)):
    cam = cv2.resize(cam , (size[0], size[1]))
    #cam = cam - cam.min()
    #cam = cam / cam.max()
    cam_min, cam_max = cam.min(), cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min)
    return cam

def calculate_multiple_iou(box_a, box_b):
    """
    Args:
        box_a: numpy.ndarray(dtype=np.int, shape=(num_a, 4))
            x0y0x1y1 convention.
        box_b: numpy.ndarray(dtype=np.int, shape=(num_b, 4))
            x0y0x1y1 convention.
    Returns:
        ious: numpy.ndarray(dtype=np.int, shape(num_a, num_b))
    """
    num_a = box_a.shape[0]
    num_b = box_b.shape[0]

    #check_box_convention(box_a, 'x0y0x1y1')
    #check_box_convention(box_b, 'x0y0x1y1')

    # num_a x 4 -> num_a x num_b x 4
    box_a = np.tile(box_a, num_b)
    box_a = np.expand_dims(box_a, axis=1).reshape((num_a, num_b, -1))

    # num_b x 4 -> num_b x num_a x 4
    box_b = np.tile(box_b, num_a)
    box_b = np.expand_dims(box_b, axis=1).reshape((num_b, num_a, -1))

    # num_b x num_a x 4 -> num_a x num_b x 4
    box_b = np.transpose(box_b, (1, 0, 2))

    # num_a x num_b
    min_x = np.maximum(box_a[:, :, 0], box_b[:, :, 0])
    min_y = np.maximum(box_a[:, :, 1], box_b[:, :, 1])
    max_x = np.minimum(box_a[:, :, 2], box_b[:, :, 2])
    max_y = np.minimum(box_a[:, :, 3], box_b[:, :, 3])

    # num_a x num_b
    area_intersect = (np.maximum(0, max_x - min_x + 1)
                      * np.maximum(0, max_y - min_y + 1))
    area_a = ((box_a[:, :, 2] - box_a[:, :, 0] + 1) *
              (box_a[:, :, 3] - box_a[:, :, 1] + 1))
    area_b = ((box_b[:, :, 2] - box_b[:, :, 0] + 1) *
              (box_b[:, :, 3] - box_b[:, :, 1] + 1))

    denominator = area_a + area_b - area_intersect
    degenerate_indices = np.where(denominator <= 0)
    denominator[degenerate_indices] = 1

    ious = area_intersect / denominator
    ious[degenerate_indices] = 0
    return ious


def resize_bbox(box, image_size, resize_size):
    """
    Args:
        box: iterable (ints) of length 4 (x0, y0, x1, y1)
        image_size: iterable (ints) of length 2 (width, height)
        resize_size: iterable (ints) of length 2 (width, height)

    Returns:
         new_box: iterable (ints) of length 4 (x0, y0, x1, y1)
    """
    check_box_convention(np.array(box), 'x0y0x1y1')
    box_x0, box_y0, box_x1, box_y1 = map(float, box)
    image_w, image_h = map(float, image_size)
    new_image_w, new_image_h = map(float, resize_size)

    newbox_x0 = box_x0 * new_image_w / image_w
    newbox_y0 = box_y0 * new_image_h / image_h
    newbox_x1 = box_x1 * new_image_w / image_w
    newbox_y1 = box_y1 * new_image_h / image_h
    return int(newbox_x0), int(newbox_y0), int(newbox_x1), int(newbox_y1)


def compute_bboxes_from_scoremaps(scoremap, scoremap_threshold_list,
                                  multi_contour_eval=False):
    """
    Args:
        scoremap: numpy.ndarray(dtype=np.float32, size=(H, W)) between 0 and 1
        scoremap_threshold_list: iterable
        multi_contour_eval: flag for multi-contour evaluation

    Returns:
        estimated_boxes_at_each_thr: list of estimated boxes (list of np.array)
            at each cam threshold
        number_of_box_list: list of the number of boxes at each cam threshold
    """
    #check_scoremap_validity(scoremap)
    height, width = scoremap.shape
    scoremap_image = np.expand_dims((scoremap * 255).astype(np.uint8), 2)

    def scoremap2bbox(threshold):
        _, thr_gray_heatmap = cv2.threshold(
            src=scoremap_image,
            thresh=int(threshold * np.max(scoremap_image)),
            maxval=255,
            type=cv2.THRESH_BINARY)
        contours = cv2.findContours(
            image=thr_gray_heatmap,
            mode=cv2.RETR_TREE,
            method=cv2.CHAIN_APPROX_SIMPLE)[_CONTOUR_INDEX]

        if len(contours) == 0:
            return np.asarray([[0, 0, 0, 0]]), 1

        if not multi_contour_eval:
            contours = [max(contours, key=cv2.contourArea)]

        estimated_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x0, y0, x1, y1 = x, y, x + w, y + h
            x1 = min(x1, width - 1)
            y1 = min(y1, height - 1)
            estimated_boxes.append([x0, y0, x1, y1])

        return np.asarray(estimated_boxes), len(contours)

    estimated_boxes_at_each_thr = []
    number_of_box_list = []
    for threshold in scoremap_threshold_list:
        boxes, number_of_box = scoremap2bbox(threshold)
        estimated_boxes_at_each_thr.append(boxes)
        number_of_box_list.append(number_of_box)

    return estimated_boxes_at_each_thr, number_of_box_list


class CamDataset(torchdata.Dataset):
    def __init__(self, scoremap_path, image_ids):
        self.scoremap_path = scoremap_path
        self.image_ids = image_ids

    def _load_cam(self, image_id):
        scoremap_file = os.path.join(self.scoremap_path, image_id + '.npy')
        return np.load(scoremap_file)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        cam = self._load_cam(image_id)
        return cam, image_id

    def __len__(self):
        return len(self.image_ids)


class LocalizationEvaluator(object):
    """ Abstract class for localization evaluation over score maps.

    The class is designed to operate in a for loop (e.g. batch-wise cam
    score map computation). At initialization, __init__ registers paths to
    annotations and data containers for evaluation. At each iteration,
    each score map is passed to the accumulate() method along with its image_id.
    After the for loop is finalized, compute() is called to compute the final
    localization performance.
    """
    # scoremap, cam_threshold_list, multi_contour_eval, gt_bbox, iou_threshold_list
    def __init__(self, dataset_name, cam_threshold_list,
                 iou_threshold_list, multi_contour_eval):
        #self.metadata = metadata
        self.cam_threshold_list = cam_threshold_list
        self.iou_threshold_list = iou_threshold_list
        self.dataset_name = dataset_name
        #self.split = split
        #self.mask_root = mask_root
        self.multi_contour_eval = multi_contour_eval

    def accumulate(self, scoremap, image_id):
        raise NotImplementedError

    def compute(self):
        raise NotImplementedError

class BoxEvaluator(LocalizationEvaluator):
    def __init__(self, **kwargs):
        super(BoxEvaluator, self).__init__(**kwargs)

        #self.image_ids = get_image_ids(metadata=self.metadata)
        self.resize_length = _RESIZE_LENGTH
        self.cnt = 0
        self.num_correct = \
            {iou_threshold: np.zeros(len(self.cam_threshold_list))
             for iou_threshold in self.iou_threshold_list}
        #self.original_bboxes = get_bounding_boxes(self.metadata)
        #self.image_sizes = get_image_sizes(self.metadata)
        #self.gt_bboxes = self._load_resized_boxes(self.original_bboxes)

    def _load_resized_boxes(self, original_bboxes):
        resized_bbox = {image_id: [
            resize_bbox(bbox, self.image_sizes[image_id],
                        (self.resize_length, self.resize_length))
            for bbox in original_bboxes[image_id]]
            for image_id in self.image_ids}
        return resized_bbox

    def accumulate(self, scoremap, gt_bbox):
        """
        From a score map, a box is inferred (compute_bboxes_from_scoremaps).
        The box is compared against GT boxes. Count a scoremap as a correct
        prediction if the IOU against at least one box is greater than a certain
        threshold (_IOU_THRESHOLD).

        Args:
            scoremap: numpy.ndarray(size=(H, W), dtype=np.float)
            image_id: string.
        """
        boxes_at_thresholds, number_of_box_list = compute_bboxes_from_scoremaps(
            scoremap=scoremap,
            scoremap_threshold_list=self.cam_threshold_list,
            multi_contour_eval=self.multi_contour_eval)
        
        boxes_at_thresholds = np.concatenate(boxes_at_thresholds, axis=0)

        multiple_iou = calculate_multiple_iou(
            np.array(boxes_at_thresholds),
            np.array([gt_bbox]))

        idx = 0
        sliced_multiple_iou = []
        for nr_box in number_of_box_list:
            sliced_multiple_iou.append(
                max(multiple_iou.max(1)[idx:idx + nr_box]))
            idx += nr_box

        for _THRESHOLD in self.iou_threshold_list:
            correct_threshold_indices = \
                np.where(np.asarray(sliced_multiple_iou) >= (_THRESHOLD/100))[0]
            self.num_correct[_THRESHOLD][correct_threshold_indices] += 1
        self.cnt += 1

    def compute(self):
        """
        Returns:
            max_localization_accuracy: float. The ratio of images where the
               box prediction is correct. The best scoremap threshold is taken
               for the final performance.
        """
        max_box_acc = []

        for _THRESHOLD in self.iou_threshold_list:
            localization_accuracies = self.num_correct[_THRESHOLD] * 100. / \
                                      float(self.cnt)
            max_box_acc.append(localization_accuracies.max())

        return max_box_acc


def evaluate_maxbox_acc(input, cls_label, bbox_label, evaluator, cams, cfg):
    """
    :param input: input tensors of the model
    :param cls_label: class label
    :param bbox_label: bounding box label
    :param logits: classification scores
    :param cams: cam of all the classes
    :param image_names: names of images
    :param cfg: configurations
    :param epoch: epoch
    :return: evaluate results
    """

    # label, top1 and top5 results
    cls_label = cls_label.tolist()

    batch = cams.shape[0]
    #image = tensor2image(input.clone().detach().cpu(), cfg.DATA.IMAGE_MEAN, cfg.DATA.IMAGE_STD)

    for b in range(batch):
        gt_bbox = bbox_label[b].strip().split(' ')
        gt_bbox = list(map(float, gt_bbox))

        # gt_known
        # mean top k
        cam_b = cams[b, [cls_label[b]], :, :]
        cam_b = torch.mean(cam_b, dim=0, keepdim=True)
        cam_b = cam_b.detach().cpu().numpy().transpose(1, 2, 0)

        # Resize and Normalize CAM
        cam_b = resize_cam(cam_b, size=(cfg.DATA.CROP_SIZE, cfg.DATA.CROP_SIZE))
        evaluator.accumulate(cam_b, gt_bbox)       
    # max_box_acc = evaluator.compute()
    # max_box_acc = np.average(max_box_acc)        
    #return max_box_acc


