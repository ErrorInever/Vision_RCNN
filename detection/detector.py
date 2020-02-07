import torch
import os
import cv2
from detection import utils
from detection.dataset import Images
from torch.utils.data import DataLoader
from tqdm import tqdm
from cls import Detect

CLASSES = utils.get_classes()


def filter_suppression(detects, treshhold):
    """
    Removes predictions which scores < treshhold
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


def draw_graphics(img, detect):
    """ drawing bounding box on image"""
    img = img.permute(1, 2, 0).cpu().numpy().copy()
    img = img * 255
    boxes = detect['boxes'].cpu()
    scores = detect['scores'].cpu().detach().numpy()
    labels = detect['labels'].cpu().detach().numpy()

    for i, bbox in enumerate(boxes):
        score = round(scores[i]*100, 1)
        label = labels[i]

        p1, p2 = tuple(bbox[:2]), tuple(bbox[2:])
        cv2.rectangle(img, p1, p2, color=(255, 0, 0), thickness=2)
        if label in CLASSES:
            text = '{cls}{prob}%'.format(cls=CLASSES[label], prob=score)
        else:
            text = '{cls}'.format(cls='unknown')
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]

        p3 = (p1[0], p1[1] - text_size[1] - 4)
        p4 = (p1[0] + text_size[0] + 4, p1[1])

        cv2.rectangle(img, p3, p4, color=(255, 0, 0), thickness=-1)
        cv2.putText(img, text, org=p1, fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1, color=(255, 255, 255), thickness=1)

    return img


class Detector(Detect):

    def __init__(self, model, device):
        super().__init__(model, device)

    def detect_on_images(self, img_path, out_path, treshhold=0.7):
        img_dataset = Images(img_path)
        dataloader = DataLoader(img_dataset, batch_size=10, num_workers=4, shuffle=False, collate_fn=utils.collate_fn)

        for images in tqdm(dataloader):
            images = list(image.to(self.device) for image in images)

            with torch.no_grad():
                detects = self.model(images)
                detects = filter_suppression(detects, treshhold)

            img_rect = []
            for i, detect in enumerate(detects):
                img_rect.append(draw_graphics(images[i], detect))

            for i, img in enumerate(img_rect):
                save_path = os.path.join(out_path, 'detection_{}.png'.format(i))
                cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    def detect_on_video(self, vid_path, out_path, treshhold=0.7):
        pass
