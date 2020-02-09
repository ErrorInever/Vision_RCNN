import torch


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
    path = '../config/classes/coco-labels-2014_2017.txt'
    num_classes = 81
    with open(path) as f:
        classes = f.read().splitlines()
        classes = dict(zip([x for x in range(1, num_classes)], classes))
    return classes

