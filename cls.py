from abc import ABC, abstractmethod


class Detect(ABC):

    def __init__(self, model, device):
        self.model = model
        self.device = device
        super().__init__()

    @abstractmethod
    def detect_on_images(self, img_path, out_path, treshhold):
        pass

    @abstractmethod
    def detect_on_video(self, vid_path, out_path, treshhold):
        pass
