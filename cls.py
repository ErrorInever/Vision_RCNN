from abc import ABC, abstractmethod


class Detect(ABC):
    """Base class of detector"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        super().__init__()

    @abstractmethod
    def detect_on_images(self, data_path, out_path, treshhold):
        pass

    @abstractmethod
    def detect_on_video(self, data_path, out_path, treshhold):
        pass
