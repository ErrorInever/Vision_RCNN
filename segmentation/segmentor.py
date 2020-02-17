import utils
import torch
from tqdm import tqdm

import visualize
from data.cls import Detect
from data.dataset import Images, Video
from torch.utils.data import DataLoader
from config.cfg import cfg


class Segmentator(Detect):
    """object segmentation"""
    def __init__(self, model, device):
        self.cls_names = utils.class_names()
        self.colors = visualize.seed_colors(self.cls_names)
        super().__init__(model, device)

    def detect_on_images(self, data_path, out_path, treshhold):
        img_dataset = Images(data_path)
        dataloader = DataLoader(img_dataset, batch_size=cfg.BATCH_SIZE, num_workers=cfg.NUM_WORKERS, shuffle=False,
                                collate_fn=utils.collate_fn)
        for images in tqdm(dataloader):
            images = list(image.to(self.device) for image in images)
            with torch.no_grad():
                predictions = self.model(images)



    def detect_on_video(self, data_path, out_path, treshhold):
        pass
