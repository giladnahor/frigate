"""Detecion network post processing.

This module will have all the functions/methods/tools needed for detection networks post
prorcessing.

Terminology:
    Det - Abbreviation for detection.
    BB - Initials for bounding box/s.
"""

import numpy as np
from collections import Counter


def merge_defused_nms_output(nms_output):
    # Defused NMS outputs two lists of equal length, so the total number of classes is *2
    num_classes = len(nms_output["nms1_d0"]) * 2
    # Create a list of length num_classes
    nms_list = [None] * num_classes
    nms_list[::2] = nms_output["nms1_d0"]  # Even class indices are from the first set
    nms_list[1::2] = nms_output["nms1_d1"]  # Odd class indices are from the second set
    return nms_list


def nms_output_to_detections(nms_output, score_threshold=0.35):
    """process batched nms output (post-infer).

    Note:
        Please use this only after nms execution in PPU.

    Args:
        nms_output(list) - the output of the post infer. Should be a list of length # of classes
        score_threshold(float) - the score threshold by which to select boxes.

    Returns:
        A dict, containg these keys:
        * boxes (obj:`np.ndarray`) -   A numpy array of length num_detections.
                           each element in the array is a numpy array of shape (detections, 4)
        * scores (obj:`np.ndarray`) -  A numpy array of length num_detections.
                           each element in the list is a float of the score
        * classes (obj:`np.ndarray`) - A numpy array of length num_detections.
                           each element in the list is an int identifying the class
        * num_detection (int) - Gives the # of detections
    """
    # Initialize the lists to pack
    boxes = []
    scores = []
    classes = []
    num_detections = 0

    for index, class_detections in enumerate(nms_output):
        # For each class, there is a numpy array of available detections
        if class_detections.size == 0:
            # Empty array means there are no detections for this class, so skip
            continue
        for detection in class_detections:
            # For each individual detection of this class
            score = detection[4]
            if score <= score_threshold:
                # If the score is beneath the threshold, then skip
                continue
            # Append the relevant info for this detection
            boxes.append(detection[:4])
            scores.append(score)
            classes.append(index + 1)
            num_detections += 1

    # Pack the lists into a dict
    detections = {
        "num_detections": num_detections,
        "boxes": np.asarray(boxes),
        "scores": np.asarray(scores),
        "classes": np.asarray(classes),
    }

    return detections


def merge_defused_nms_output_tf_api(nms_output):
    if len(nms_output["nms1_d0"].shape) == 4:
        nms_output["nms1_d0"] = nms_output["nms1_d0"].squeeze()
        nms_output["nms1_d1"] = nms_output["nms1_d1"].squeeze()
    classes, reg, max_outputs = nms_output["nms1_d0"].shape
    nms = np.zeros([classes * 2, reg, max_outputs], dtype=np.float32)
    nms[::2, :, :] = nms_output["nms1_d0"]
    nms[1::2, :, :] = nms_output["nms1_d1"]
    return nms


def nms_output_to_detections_tf_api(nms_output, score_threshold=0.35):
    """process batched nms output (post-infer).

    Note:
        Please use this only after nms execution in PPU.

    Args:
        nms_output(np.array) - the output of the post infer. should be of size(batch, classes, 5, max_proposals)
        score_threshold(float) - the score threshold by which to select boxes.

    Returns:
        A dict, containg these keys:
        * boxes (list) -   A list of length batches.
                           each element in the list is a np array of shape (detections, 4)
        * scores (list) -  A list of length batches.
                           each element in the list is a np array of shape (detections,)
        * classes (list) - A list of length batches.
                           each element in the list is a np array of shape (detections,)
        * num_detection (obj:`np.ndarray`) - Each element gives the #detections in each batch
    """
    if len(nms_output.shape) < 4:
        # expand for 'batch' annotation
        nms_output = np.expand_dims(nms_output, axis=0)

    indices = np.argwhere(nms_output[:, :, 4, :] > score_threshold)
    boxes = nms_output[indices[:, 0], indices[:, 1], :4, indices[:, 2]]
    scores = nms_output[indices[:, 0], indices[:, 1], 4, indices[:, 2]]
    classes = indices[:, 1] + 1
    batch = indices[:, 0]

    d = Counter(batch)

    batch_size = nms_output.shape[0]
    idx = np.zeros(shape=(batch_size - 1), dtype=np.uint16)
    num_detections = np.zeros(shape=batch_size, dtype=np.uint16)

    last_ind = 0

    for i in range(batch_size - 1):
        if i in d.keys():
            last_ind += d[i]
            num_detections[i] = d[i]
            idx[i] = last_ind

            num_detections[batch_size - 1] = (
                0 if not batch_size - 1 in d else d[batch_size - 1]
            )

            classes = np.split(classes, idx)
            boxes = np.split(boxes, idx)
            scores = np.split(scores, idx)
    scores -= 0.01

    if batch_size == 1:
        num_detections = len(boxes)

    detections = {
        "num_detections": num_detections,
        "boxes": boxes,
        "scores": scores,
        "classes": classes,
    }

    return detections


# Detection post process classes
class DetectionPostNMS:
    def __init__(
        self,
        score_threshold=0.4,
        defuse_nms=True,
        use_tf_nms_format=True,
        *args,
        **kwargs
    ):
        self.score_threshold = score_threshold
        self._defuse_nms = defuse_nms
        self._use_tf_nms_format = use_tf_nms_format

    def run(self, nms_output, score_threshold=None):
        if self._defuse_nms:
            if self._use_tf_nms_format:
                nms_output = merge_defused_nms_output_tf_api(nms_output)
            else:
                nms_output = merge_defused_nms_output(nms_output)
        if score_threshold is None:
            score_threshold = self.score_threshold
        if self._use_tf_nms_format:
            detections = nms_output_to_detections_tf_api(
                nms_output, score_threshold=score_threshold
            )
        else:
            detections = nms_output_to_detections(
                nms_output, score_threshold=score_threshold
            )
        return detections
