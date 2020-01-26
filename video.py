import cv2
import os
from datetime import datetime
import utils


# TODO: release decorator to track time
# TODO: release batchs frames
class Video:
    """ Defines a video"""

    def __init__(self, video_path, save_path):
        """
        :param video_path: path to a video file
        :param save_path: path to output directory
        """
        self.cap = cv2.VideoCapture(video_path)
        self.video_path = video_path
        self.save_path = os.path.join(save_path, 'detection_{}.avi'.format(datetime.today().strftime('%Y-%m-%d')))
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.width, self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                                  int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.__len__() / self.fps
        self.out = cv2.VideoWriter(self.save_path, self.fourcc, self.fps, (self.width, self.height))

    def __str__(self):
        info = 'duration: {}\nframes: {}\nresolution: {}x{}\nfps: {}'.format(round(self.duration, 1),
                                                                             self.__len__(),
                                                                             self.width, self.height,
                                                                             round(self.fps, 1))
        return info

    def __len__(self):
        """ total number of frames"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame(self):
        """convert frame to tensor and return object of generator frame by frame"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = utils.frame_to_tensor(frame)
                yield frame
