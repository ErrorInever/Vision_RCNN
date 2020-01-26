import utils
import torch
import os
import cv2
from images import Images
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm


def filter_suppression(detects, treshhold):
    """
    Remove boxes,labels which score < treshhold
    :param detects: list of dictionary
    :param treshhold: float
    :return: list of dictionary
    """
    samples = []

    for detect in detects:
        scores = detect['scores']
        mask = len(list(filter(lambda x: x >= treshhold, scores)))

        sample = {'boxes': detect['boxes'][:mask],
                  'labels': detect['labels'][:mask],
                  'scores': detect['scores'][:mask]
                  }
        samples.append(sample)

    return samples


def filter_labels(labels):
    mask = utils.get_classes()
    classes = []
    for label in labels:
        classes.append(mask[label])

    return classes


def draw_box(img, detect):
    """ drawing bounding box on image"""
    img = img.permute(1, 2, 0).cpu().numpu().copy()
    img = img * 255
    boxes = detect['boxes'].cpu()
    scores = detect['scores'].cpu().detach().numpy()
    classes = detect['labels'].cpu().detach().numpy()

    for i, box in enumerate(boxes):
        score = round(scores[i] * 100, 1)

    # TODO



class Detector:

    def __init__(self, model, device, treshhold=0.7):
        self.model = model
        self.device = device
        self.treshhold = treshhold

    def detect_on_images(self, img_path, out_path):
        img_dataset = Images(img_path)
        dataloader = DataLoader(img_dataset, batch_size=10, num_workers=4, shuffle=False, collate_fn=utils.collate_fn)

        for images in tqdm(dataloader):
            images = list(image.to(self.device) for image in images)

            with torch.no_grad():
                detects = self.model(images)
                detects = filter_suppression(detects, self.treshhold)

            for i, detect in enumerate(detects):
                # TODO: draw boxes
                pass

            # TODO: save images

    def detect_on_video(self):
        pass
