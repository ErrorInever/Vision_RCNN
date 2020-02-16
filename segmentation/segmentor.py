import utils
from data.cls import Detect


class Segmentator(Detect):
    """object segmentation"""
    def __init__(self, model, device):
        self.cls_names = utils.class_names()
        self.colors = utils.seed_colors(self.cls_names)
        super().__init__(model, device)

    def detect_on_images(self, data_path, out_path, treshhold):
        pass

    def detect_on_video(self, data_path, out_path, treshhold):
        pass
