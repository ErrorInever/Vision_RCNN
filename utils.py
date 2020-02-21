import torch
import numpy as np
from config.classes.coco_labels import CLASS_NAMES


def img_to_tensor(image):
    """
    Converts image to tensor and normalize
    :param image: ``numpy_array [H, W, 3]``
    :return: ``tensor(3, H, W)``
    """
    image = torch.from_numpy(image).float() / 255.0
    image = image.permute(2, 0, 1)
    return image


def reverse_normalization(tensor):
    """
    Converts image tensor to numpy array
    :param tensor: ``tensor(3, H, W)``
    :return: ``numpy_array [H, W, 3]``
    """
    return tensor.permute(1, 2, 0).mul(255).cpu().byte().numpy()


def flip_vert_tensor(tensor):
    """Vertical flip 3R tensor. EXPENSIVE OPERATION"""
    return tensor.flip(2, 1)


def collate_fn(batch):
    """Pack batch"""
    return tuple(batch)


def class_names():
    """
    Get coco class names
    :return ``Dict{class_id : "class_name"}``
    """
    return CLASS_NAMES


def filter_prediction(predictions, treshhold):
    """
    Removes prediction where scores < threshold
    :param predictions:
    ``List[Dict[Tensor]]`` The fields of the ``Dict`` are as follows:
        - boxes (``FloatTensor[N, 4]``)
        - labels (``Int64Tensor[N]``)
        - scores (``Tensor[N]``)
        * optional
        - masks (``UInt8Tensor[N, 1, H, W]``)
    :param treshhold: ``float``, threshold
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
            sample['masks'] = prediction['masks'][:tresh]

        samples.append(sample)

    return samples
