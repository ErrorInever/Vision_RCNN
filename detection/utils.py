import torch
from config.classes.coco_labels import CLASS_NAMES


def frame_to_tensor(frame):
    """
    Convert image frame to tensor
    :param frame: img
    :return: 3R tensor
    """
    frame = torch.from_numpy(frame).float() / 255.0
    frame = frame.permute(2, 0, 1)
    return frame


def collate_fn(batch):
    return tuple(batch)


def class_names():
    """
    Get coco class names
    :return dictionary {id : name}"""
    return CLASS_NAMES


