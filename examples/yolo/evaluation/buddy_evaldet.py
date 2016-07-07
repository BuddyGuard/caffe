import numpy as np
from PIL import Image


class GTObjects(object):
    """
    Class for storing parameters of given class objects, and the
    detection status in a single image
    """

    def __init__(self):
        self.bboxes = []
        self.difficult = []
        self.det = []


def filter_class_specific_objects(objs_dict, class_name):
    """ Get all objects related to given class
    TODO: filter objects that are difficult
    """
    npos = 0
    gt_objects = dict()
    # print("dict_length",len(objs_dict.keys()))
    for gtid in objs_dict.keys():
        gt_obj = GTObjects()
        obj = objs_dict[gtid]
        names = np.array(obj["class_names"])
        class_idxs = np.where(names == class_name)[0]
        # print(class_idxs)
        gt_obj.bboxes = np.array(obj["bboxes"])[class_idxs]
        # gt_obj.difficult = np.array(obj.difficult)[class_idxs]
        gt_obj.det = np.zeros(len(class_idxs))
        gt_objects[gtid] = gt_obj
        # Consider only non difficult objects
        npos += len(class_idxs)  # len(gt_obj.difficult) - np.count_nonzero(gt_obj.difficult)

    return gt_objects, npos


def load_detection_results(file_path):
    """ Load detection results of given class"""
    # file_path = self.__init_params.detrespath.format(model_name, class_name)
    results = np.genfromtxt(file_path, dtype=np.str, delimiter=" ")
    # print(file_path)
    ids = results[:, 0]
    # print(ids)
    confidence = results[:, 1]
    bboxes = results[:, 2:6]

    # cast to float
    confidence = confidence.astype(np.float)
    bboxes = bboxes.astype(np.float)

    # sort indices based on confidence scores in decreasing order
    desc_order_idxs = np.argsort(confidence)[::-1]

    # rearrange confidence,  ids and bbboxes
    ids = ids[desc_order_idxs]
    confidence = confidence[desc_order_idxs]
    bboxes = bboxes[desc_order_idxs]

    return ids, confidence, bboxes


def load_ground_truth(file_path):
    """ Load ground truth from a buddynet txt label file"""
    # file_path = self.__init_params.detrespath.format(model_name, class_name)
    import os
    if not os.path.isfile(file_path):
        print("not path %s" % file_path)
        return np.array([]), np.array([])
    results = np.genfromtxt(file_path, dtype=np.str)
    # print(file_path)
    # print(results)
    # sys.stdout.flush()
    classes = np.atleast_2d(results)[:, 0]
    bboxes = np.atleast_2d(results)[:, 1:5]

    # cast to float
    classes = classes.astype(np.integer)
    bboxes = bboxes.astype(np.float)
    # sort indices based on class labels in decreasing order
    desc_order_idxs = np.argsort(classes)[::-1]

    # rearrange the classes and bbboxes
    classes = classes[desc_order_idxs]
    bboxes = bboxes[desc_order_idxs]

    return classes, bboxes


def assign_detections_to_gtobjects(ids, bboxes, gt_objects, min_overlap=0.5):
    """ Compute overlap area and estimate true and false positives"""
    true_positives = np.array([0] * len(ids))
    false_positives = np.array([0] * len(ids))
    import sys
    for i, idx in enumerate(ids):
        # print(idx)
        sys.stdout.flush()
        bbox = bboxes[i]
        gt_obj = gt_objects[idx]

        overlap_max = 0
        pos_overlap_max = 0
        # compute overlap area
        inside = False
        for j, gt_bbox in enumerate(gt_obj.bboxes):
            new_mins = np.maximum.reduce([bbox[:2], gt_bbox[:2]])
            new_maxs = np.minimum.reduce([bbox[2:4], gt_bbox[2:4]])
            new_bbox = np.append(new_mins, new_maxs)
            # compute width and height of bounding box
            width = new_bbox[2] - new_bbox[0] + 1
            height = new_bbox[3] - new_bbox[1] + 1

            if (width > 0) and (height > 0):
                # compute overlap as area of intersection / area of union
                area_union = ((bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1) +
                              (gt_bbox[2] - gt_bbox[0] + 1) * (gt_bbox[3] - gt_bbox[1] + 1) -
                              (width * height))
                overlap = width * height / area_union
                # update maximum overlap area
                if overlap > overlap_max:
                    overlap_max = overlap
                    pos_overlap_max = j
                    inside = False
                    if (
                        new_bbox == gt_bbox).all():  # gt_bbox is inside the detected area, the gt_bbox size might be small
                        inside = True  # relax the min_overlap for this case

        # Estimate true and false positives
        if overlap_max >= min_overlap or (inside and overlap_max >= min_overlap * 0.5):
            if not gt_obj.det[pos_overlap_max]:
                true_positives[i] = 1
                gt_obj.det[pos_overlap_max] = True
            else:
                false_positives[i] = 1
        else:
            false_positives[i] = 1

    return true_positives, false_positives


def compute_precision_recall(number_of_class_objs, true_positives, false_positives):
    """ Compute precision recall"""
    tp_cumsum = np.cumsum(true_positives)
    fp_cumsum = np.cumsum(false_positives)
    # print(number_of_class_objs)
    recall = 1.0 * tp_cumsum / number_of_class_objs
    precision = 1.0 * tp_cumsum / (tp_cumsum + fp_cumsum)

    return recall, precision


def compute_avg_precision(recall, precision):
    """ Compute average precision"""
    mrec = np.append([0], recall)
    mrec = np.append(mrec, [1])

    mpre = np.append([0], precision)
    mpre = np.append(mpre, [0])

    for pos in xrange(len(mpre) - 2, -1, -1):
        mpre[pos] = max(mpre[pos], mpre[pos + 1])

    idxs = [i + 1 for i, (mrec1, mrec2) in enumerate(zip(mrec[1:], mrec[:-1])) if mrec1 != mrec2]
    idxs = np.array(idxs)

    ap = np.sum((mrec[idxs] - mrec[idxs - 1]) * mpre[idxs])

    return ap


def evaluate(pred_path, objs_dict, class_id, min_overlap):
    gt_objects, npos = filter_class_specific_objects(objs_dict, class_id)
    print("npos", npos)
    ids, confidence, bboxes = load_detection_results(pred_path)
    # print(ids)
    # print("ids",len(ids))
    true_positives, false_positives = assign_detections_to_gtobjects(ids, bboxes, gt_objects, min_overlap=min_overlap)
    # print(false_positives)
    # print(np.cumsum(false_positives))
    # print(true_positives)
    # print(np.cumsum(true_positives))
    print("cumsum_tp,cumsum_fp", np.cumsum(true_positives), np.cumsum(false_positives))
    recall, precision = compute_precision_recall(npos, true_positives, false_positives)
    # print(recall)
    # print(precision)
    # print("dodo")
    ap = compute_avg_precision(recall, precision)
    return true_positives, false_positives, recall, precision, ap


def read_val_set(val_path):
    ids = []
    im_whs = []
    with open(val_path, "r") as val_f:
        for jpeg in val_f:
            path = jpeg.strip()
            if path != "":
                pass
            else:
                continue
            im = Image.open(path)
            width, height = im.size
            im_whs.append((width, height))
            ids.append(path[path.rfind("/") + 1:-4])
        return np.array(ids), im_whs


def denormalize_bbox(bbox, img_w, img_h):
    """ denormalize according to the image width and height """
    [bx, by, bw, bh] = bbox
    nw = bw * float(img_w)
    nh = bh * float(img_h)
    nx = bx * img_w - nw / 2.0
    ny = by * img_h - nh / 2.0
    return [nx, ny, nw, nh]


def denormalize_bbox_xy_from_wh(bbox, w, h):
    [x, y, bw, bh] = denormalize_bbox(bbox, w, h)
    return [x, y, x + bw, y + bh]


def denormalize_xy_from_wh(x, y, bw, bh, w, h):
    left = (x - bw / 2) * w
    right = (x + bw / 2) * w
    top = (y - bh / 2) * h
    bot = (y + bh / 2) * h
    return [left, right, top, bot]


def normalize_bbox_obj(bbox, obj):
    """ bbox conversion directly from objects """
    return normalize_bbox_from_wh(bbox, obj["width"], obj["height"])


def normalize_bbox_from_wh(bbox, img_w, img_h):
    """ buddynet/darknet normalized bounding box format (where everything is between 0.0 and 1.0) """
    [x, y, w, h] = bbox
    bx = (x + w / 2.0) / img_w
    by = (y + h / 2.0) / img_h
    bw = w / float(img_w)
    bh = h / float(img_h)
    return [bx, by, bw, bh]


def normalize_bbox_from_xy(bbox, img_w, img_h):
    """ buddynet/darknet normalized bounding box format (where everything is between 0.0 and 1.0) """
    [x, y, x_max, y_max] = bbox
    bbox = [x, y, x_max - x, y_max - y]
    return normalize_bbox_from_wh(bbox, img_w, img_h)


if __name__ == "__main__":
    import argparse

    """ evaluation of data with PASCAL VOC bboxes [xmin,ymin,xmax,ymax] """
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions",
                        help="folder of the the prediction results are recorded in seperate txt files for each category \ne.g. path/to/VOCdevkit/results/VOC2012/Main/",
                        type=str)
    parser.add_argument("validation", help="the txt file containing the list of full jpg paths \ne.g. path/to/val.txt",
                        type=str)
    parser.add_argument("labels", help="folder of the buddynet label txt files \ne.g. path/to/labels/")
    parser.add_argument("denormalize", help="0 or 1, if the labels contain normalized bboxes, this should be 1")
    parser.add_argument("reduced",
                        help="0 or 1, if reduced the class labels are cat:0, dog:1, person:2 else cat:7, dog:11, person:14")
    parser.add_argument("min_overlap", help="between 0 and 1, minimum overlap for the evaluation")
    args = parser.parse_args()

    base = args.predictions
    val_path = args.validation
    labels_path = args.labels
    denormalize = args.denormalize == "1"
    reduced = args.reduced
    min_overlap = float(args.min_overlap)
    val_ids, im_whs = read_val_set(val_path)
    # ground_truth={0:[],1:[],2:[]}
    objs_dict = {}

    for i, wh in zip(val_ids, im_whs):
        classes, bboxes = load_ground_truth(labels_path + i + ".txt")
        # if loaded from normalized labels use below instead
        if denormalize:  # from normalized labels txt bbox to pascal bbox
            bboxes = np.array([denormalize_bbox_xy_from_wh(bbox, *wh) for bbox in bboxes])
        else:
            bboxes = np.array(bboxes)
        obj = {"class_names": classes, "bboxes": bboxes}  # TODO add difficulty??
        # print(classes)
        objs_dict[i] = obj
    # raw_input("continue?")
    aps = 0
    labels=["person","cat","dog"]
    if reduced=="1":
        class_ids = [2, 0, 1]
    elif reduced=="0":
        class_ids = [14, 7, 11]
    elif reduced=="pet":
        class_ids=[3]
        labels=["pet"]
    else:
        label_map={"person":2,"cat":0,"dog":1}
        class_ids=[label_map[reduced]]
        labels=[reduced]
    # gt_objects=pd.DataFrame(ground_truth,index=val_ids)
    for i, label in zip(class_ids, labels):
        true_positives, false_positives, recall, precision, ap = evaluate(base + "yolo-tiny_det_val_%s.txt" % label,
                                                                          objs_dict, i, min_overlap=min_overlap)
        print(label, ap)
        aps += ap
    print(aps / 3.0)
