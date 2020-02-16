import cv2
from config.cfg import cfg


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
        cv2.rectangle(img, p1, p2, color=colors[label], thickness=cfg.THICKNESS_BBOX)
        text = '{cls} {prob}%'.format(cls=cls_names[label], prob=score)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, cfg.TEXT_SIZE, 1)[0]

        p3 = (p1[0] - 2, p1[1] - text_size[1] - 6)
        p4 = (p1[0] + text_size[0] + 4, p1[1])

        cv2.rectangle(img, p3, p4, color=colors[label], thickness=-1)
        cv2.putText(img, text, (p1[0], p1[1] - text_size[1] + 6), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=cfg.TEXT_SIZE, color=(0, 0, 0), thickness=1)
    return img


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
