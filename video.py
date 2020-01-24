import cv2
import os
from datetime import datetime
import utils


# class Converter:
#     """Converting video frames to images, and vice versa"""
#     @staticmethod
#     def get_frame(self, video):
#         if not isinstance(video, Video):
#             raise TypeError('get wrong type: {}'.format(type(video)))
#         else:
#             while video.cap.isOpened():
#                 ret, frame = video.cap.read()
#                 if ret:
#                     yield frame
#
#     def images_to_video(self, images):
#         pass

# TODO: release decorator to track time
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
        self.out = cv2.VideoWriter(self.save_path, self.fourcc, (self.width, self.height))

    def __str__(self):
        info = 'duration:{}: frames:{}|resolution:{}x{}|fps:{}'.format(self.duration, self.__len__(),
                                                                       self.width, self.height, self.fps)
        return info

    def __len__(self):
        """ total number of frame"""
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_frame(self):
        """return generator frame by frame"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = utils.frame_to_tensor(frame)
                yield frame
