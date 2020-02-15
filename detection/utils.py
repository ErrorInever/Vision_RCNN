import torch
import numpy as np
from config.classes.coco_labels import CLASS_NAMES


def frame_to_tensor(frame):
    """
    convert frame to tensor
    :param frame: frame of video
    :return: 3R tensor
    """
    frame = torch.from_numpy(frame).float() / 255.0
    frame = frame.permute(2, 0, 1)
    # TODO: release batchs
    # FIXME: possible need return list of frame
    return frame


def collate_fn(batch):
    return tuple(batch)


def class_names():
    """:return class names"""
    return CLASS_NAMES


def color_bounding_box(classes):
    """ :return array of colors"""
    return np.random.uniform(0, 255, size=(len(classes), 3))
