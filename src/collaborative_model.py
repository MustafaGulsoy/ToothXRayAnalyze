import numpy as np


def combine_and_refine_results(segmentation_results_mask_rcnn, segmentation_results_unet,
                               identification_results_faster_rcnn, identification_results_yolo_v5):
    combined_results = []
    for i in range(len(segmentation_results_mask_rcnn)):
        combined_result = {
            'mask_rcnn': segmentation_results_mask_rcnn[i],
            'unet': segmentation_results_unet[i],
            'faster_rcnn': identification_results_faster_rcnn[i],
            'yolo_v5': identification_results_yolo_v5[i]
        }
        combined_results.append(combined_result)

    refined_results = []
    for result in combined_results:
        nms_results = non_max_suppression(result['faster_rcnn'])
        numbered_teeth = number_teeth(nms_results, result['mask_rcnn'])
        refined_segmentation = filter_out_of_bounds(numbered_teeth, result['unet'])
        refined_results.append(refined_segmentation)

    return refined_results


def non_max_suppression(detections, threshold=0.7):
    if len(detections) == 0:
        return []

    detections = sorted(detections, key=lambda x: x[4], reverse=True)
    keep = []

    while detections:
        max_detection = detections.pop(0)
        keep.append(max_detection)

        detections = [d for d in detections if intersection_over_union(max_detection, d) < threshold]

    return keep


def intersection_over_union(box1, box2):
    x1, y1, x2, y2 = box1[:4]
    x1_, y1_, x2_, y2_ = box2[:4]

    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)

    inter_area = max(0, xi2 - xi1 + 1) * max(0, yi2 - yi1 + 1)

    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_ - x1_ + 1) * (y2_ - y1_ + 1)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou


def number_teeth(detections, mask_rcnn_results):
    teeth_centers = []
    for detection in detections:
        x1, y1, x2, y2 = detection[:4]
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        teeth_centers.append((center_x, center_y))

    numbered_teeth = assign_iso_numbers(teeth_centers, mask_rcnn_results)
    return numbered_teeth


def assign_iso_numbers(teeth_centers, mask_rcnn_results):
    iso_teeth_numbers = {}
    for idx, center in enumerate(teeth_centers):
        iso_teeth_numbers[f'Tooth_{idx + 1}'] = center
    return iso_teeth_numbers


def filter_out_of_bounds(teeth_centers, unet_results):
    filtered_results = []
    for tooth, center in teeth_centers.items():
        if is_within_bounds(center, unet_results):
            filtered_results.append((tooth, center))
    return filtered_results


def is_within_bounds(center, unet_results):
    x, y = center
    return 0 <= x < unet_results.shape[1] and 0 <= y < unet_results.shape[0]
