import torch
import numpy as np


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
