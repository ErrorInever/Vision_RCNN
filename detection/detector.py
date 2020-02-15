import torch
import os
import cv2
import numpy as np
from datetime import datetime
from detection import utils
from detection.dataset import Images, Video
from torch.utils.data import DataLoader
from tqdm import tqdm
from cls import Detect

# TODO: refactoring, color classes, dynamic text size


def filter_threshold(detects, threshold):
    """
    Removes predictions which scores < treshhold
    :param detects: list of dictionary
    [{boxes: [[x1, y1, x2, y2], ...],
     scores: [float],
     labels: [int],
     , ...]
    :param threshold: float
    :return: list of dictionary
    """
    samples = []

    for detect in detects:
        scores = detect['scores']
        mask = len(list(filter(lambda x: x >= threshold, scores)))

        sample = {'boxes': detect['boxes'][:mask],
                  'labels': detect['labels'][:mask],
                  'scores': detect['scores'][:mask]
                  }
        samples.append(sample)

    return samples


def draw_bbox(img, prediction, cls_names, colors):
    """ draws bounding boxes, scores and classes on image
    :param img: tensor
    :param prediction: dictionary
    {boxes: [[x1, y1, x2, y2], ...],
     scores: [float],
     labels: [int]}
    :param cls_names: dictionary class names
    :param colors: numpy array of colors
    """
    img = img.permute(1, 2, 0).cpu().numpy().copy()
    img = img * 255
    boxes = prediction['boxes'].cpu()
    scores = prediction['scores'].cpu().detach().numpy()
    labels = prediction['labels'].cpu().detach().numpy()

    for i, bbox in enumerate(boxes):
        score = round(scores[i]*100, 1)
        label = labels[i]
        p1, p2 = tuple(bbox[:2]), tuple(bbox[2:])
        cv2.rectangle(img, p1, p2, color=colors[label], thickness=3)
        text = '{cls} {prob}%'.format(cls=cls_names[label], prob=score)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

        p3 = (p1[0], p1[1] - text_size[1] - 4)
        p4 = (p1[0] + text_size[0] + 4, p1[1])

        cv2.rectangle(img, p3, p4, color=colors[label], thickness=-1)
        cv2.putText(img, text, org=p1, fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1, color=(0, 0, 0), thickness=1)
    return img


def get_all_time(func):
    """get function execution time"""
    def wrapped(*args, **kwargs):
        start_time = datetime.now()
        res = func(*args, **kwargs)
        print('wasted time = {}'.format((datetime.now() - start_time).total_seconds()))
        return res
    return wrapped


class Detector(Detect):
    """object detector"""
    def __init__(self, model, device):
        """
        :param model: instance of net
        :param device: can be cpu or cuda device
        """
        self.cls_names = utils.class_names()
        self.colors = utils.color_bounding_box(self.cls_names)
        super().__init__(model, device)

    @get_all_time
    def detect_on_images(self, img_path, out_path, threshold=0.7):
        """
        Detects objects on images and saves it
        :param img_path: path to images data
        :param out_path: path to output results
        :param threshold: threshold detection
        """
        img_dataset = Images(img_path)
        dataloader = DataLoader(img_dataset, batch_size=10, num_workers=4, shuffle=False, collate_fn=utils.collate_fn)

        for images in tqdm(dataloader):
            images = list(image.to(self.device) for image in images)

            with torch.no_grad():
                predictions = self.model(images)

            predictions = filter_threshold(predictions, threshold)

            img_rect = []
            for i, predict in enumerate(predictions):
                img_rect.append(draw_bbox(images[i], predict, self.cls_names, self.colors))

            for i, img in enumerate(img_rect):
                save_path = os.path.join(out_path, 'detection_{}.png'.format(i))
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    @get_all_time
    def detect_on_video(self, data_path, out_path, threshold=0.7):
        """
        Detects objects on video and saves it
        :param data_path: path to video
        :param out_path: path to output result
        :param threshold: threshold detection
        """
        video = Video(data_path, out_path)
        print(video)

        for frames in tqdm(video.get_frame(), total=len(video)):
            frames = [frame.to(self.device) for frame in frames]

            with torch.no_grad():
                predictions = self.model(frames)

            predictions = filter_threshold(predictions, threshold)

            img_rect = []
            for i, predict in enumerate(predictions):
                img_rect.append(draw_bbox(frames[i], predict, self.cls_names, self.colors))

            for i, img in enumerate(img_rect):
                img = np.uint8(img)
                video.out.write(img)
        video.out.release()
        print('Done. Detect video saves to {}'.format(video.save_path))
