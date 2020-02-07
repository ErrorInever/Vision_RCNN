import torch
import cv2
import os


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


def get_classes():
    """ return dictionary of classes"""
    PATH = 'coco2017_classes/coco-labels-2014_2017.txt'
    with open(PATH) as f:
        classes = f.read().splitlines()
        classes = dict(zip([x for x in range(1, 81)], classes))
    return classes

