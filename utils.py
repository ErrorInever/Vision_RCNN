import torch
import numpy as np
from config.classes.coco_labels import CLASS_NAMES


def img_to_tensor(image):
    """
    Converts image to tensor and normalize
    :param image: image numpy array format
    :return: 3R tensor
    """
    image = torch.from_numpy(image).float() / 255.0
    image = image.permute(2, 0, 1)
    return image


def reverse_normalization(tensor):
    """
    Converts image tensor to numpy array
    :param tensor: image 3R tensor format
    :return: image numpy array format
    """
    return tensor.permute(1, 2, 0).mul(255).cpu().numpy().astype(np.uint8).copy()


def flip_vert_tensor(tensor):
    """vertical flip 3R tensor. EXPENSIVE OPERATION"""
    return tensor.flip(2, 1)


def collate_fn(batch):
    """ pack batch"""
    return tuple(batch)


def class_names():
    """
    Get coco class names
    :return dictionary {id : name}
    """
    return CLASS_NAMES


def filter_prediction(predictions, treshhold):
    """
    Remove prediction where scores < threshold
    :param predictions:
    ``List[Dict[Tensor]]`` The fields of the ``Dict`` are as follows:
        - boxes (``FloatTensor[N, 4]``)
        - labels (``Int64Tensor[N]``)
        - scores (``Tensor[N]``)
        * optional
        - masks (``UInt8Tensor[N, 1, H, W]``)
    :param treshhold: threshold
    :return: the same predictions without scores < threshold
    """
    samples = []
    for i, prediction in enumerate(predictions):
        tresh = len([x for x in predictions[i]['scores'] if x >= treshhold])
        sample = {
            'boxes': prediction['boxes'][:tresh],
            'labels': prediction['labels'][:tresh],
            'scores': prediction['scores'][:tresh],
        }

        if 'masks' in prediction:
            sample = {
                'masks': prediction['masks'][:tresh]
            }
        samples.append(sample)

    return samples


# def filter_threshold(detects, threshold):
#     """
#     Removes predictions which scores < treshhold
#     :param detects: list of dictionary
#     [{boxes: [[x1, y1, x2, y2], ...],
#      scores: [float],
#      labels: [int],
#      , ...]
#     :param threshold: float
#     :return: list of dictionary
#     """
#     samples = []
#
#     for detect in detects:
#         scores = detect['scores']
#         mask = len(list(filter(lambda x: x >= threshold, scores)))
#
#         sample = {'boxes': detect['boxes'][:mask],
#                   'labels': detect['labels'][:mask],
#                   'scores': detect['scores'][:mask]
#                   }
#         samples.append(sample)
#
#     return samples