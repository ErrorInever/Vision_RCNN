import torch
import os
from detection import utils
import math
import cv2
import numpy as np
from datetime import datetime
import visualize
from data.dataset import Images, Video
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.cls import Detect
from visualize import display_objects
from config.cfg import cfg


def execution_time(func):
    """Print wasted time of function"""
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
        self.colors = visualize.assign_colors(self.cls_names)
        super().__init__(model, device)

    @execution_time
    def detect_on_images(self, img_path, out_path, display_masks=True, display_boxes=True, display_caption=True):
        """
        Detects objects on images and saves it
        :param display_caption: if true - displays caption on image
        :param display_boxes: if true - displays boxes on image
        :param display_masks: if true - displays masks on image
        :param img_path: path to images data
        :param out_path: path to output results
        """
        img_dataset = Images(img_path)
        dataloader = DataLoader(img_dataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, shuffle=False,
                                collate_fn=utils.collate_fn)

        for images in tqdm(dataloader):
            images = list(image.to(self.device) for image in images)

            with torch.no_grad():
                predictions = self.model(images)

            images = display_objects(images, predictions, self.cls_names, self.colors,
                                     display_masks=display_masks,
                                     display_boxes=display_boxes,
                                     display_caption=display_caption,
                                     score_threshold=cfg.SCORE_THRESHOLD)

            for i, img in enumerate(images):
                save_path = os.path.join(out_path, 'detection_{}.png'.format(i))
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    @execution_time
    def detect_on_video(self, data_path, out_path, threshold,
                        display_masks=True, display_boxes=True, display_caption=True, flip=False):
        """
        Detects objects on video and saves it
        :param display_caption: if true - displays caption on video
        :param display_boxes: if true - displays boxes on video
        :param display_masks: if true - displays masks on video
        :param flip: if true - flip video
        :param data_path: path to video
        :param out_path: path to output result
        :param threshold: threshold detection
        """
        video = Video(data_path, out_path, flip)
        # FIXME: num_workers makes infinity loop
        dataloader = DataLoader(video, batch_size=cfg.BATCH_SIZE, collate_fn=utils.collate_fn)

        for batch in tqdm(dataloader, total=(math.ceil(len(dataloader) / cfg.BATCH_SIZE))):
            images = [frame.to(self.device) for frame in batch]
            with torch.no_grad():
                predictions = self.model(images)

            images = display_objects(images, predictions, self.cls_names, self.colors,
                                     display_masks=display_masks,
                                     display_boxes=display_boxes,
                                     display_caption=display_caption,
                                     score_threshold=cfg.SCORE_THRESHOLD)

            for i, img in enumerate(images):
                video.out.write(img.astype(np.uint8))
        video.out.release()
        print('Done. Detect on video saves to {}'.format(video.save_path))
