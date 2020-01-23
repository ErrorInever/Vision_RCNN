import cv2
import os

from datetime import datetime


class Video:

    def __init__(self, video_path, save_path):
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
