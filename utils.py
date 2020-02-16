import numpy as np
import torch
from config.classes.coco_labels import CLASS_NAMES


def frame_to_tensor(frame):
    """
    Convert frame to tensor
    :param frame: img
    :return: 3R tensor
    """
    frame = torch.from_numpy(frame).float() / 255.0
    frame = frame.permute(2, 0, 1)
    return frame


def flip_vert_tensor(tensor):
    """vertical flip"""
    return tensor.flip(2, 1)


def collate_fn(batch):
    return tuple(batch)


def class_names():
    """
    Get coco class names
    :return dictionary {id : name}"""
    return CLASS_NAMES


def seed_colors(classes):
    colors = np.random.uniform(80, 255, size=(len(classes), 3))
    # bio
    colors[1] = [204, 6, 5]       # person
    colors[16] = [118, 255, 122]  # bird
    colors[17] = [229, 81, 55]    # cat
    colors[18] = [219, 215, 210]  # dog
    # vehicle
    colors[2] = [0, 149, 182]     # bicycle
    colors[3] = [127, 255, 212]   # car
    colors[4] = [205, 164, 222]   # motorcycle
    colors[5] = [249, 132, 229]   # airplane
    colors[6] = [248, 243, 43]    # bus
    colors[7] = [100, 149, 237]   # train
    colors[8] = [222, 76, 138]    # truck
    # another
    colors[10] = [237, 118, 14]   # traffic light
    return colors


def random_colors(classes):
    """
    Each execution makes different colors
    :return numpy array"""
    return np.random.uniform(80, 255, size=(len(classes), 3))