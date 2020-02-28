import numpy as np
import cv2
import logging
from config.cfg import cfg
from PIL import Image, ImageDraw, ImageFont
from detection import utils

logger = logging.getLogger(__name__)


def random_colors(classes):
    """
    Each execution makes random colors
    :param classes: ``Dict[class_id, "name of class"]``, dictionary of class names
    :return ``Tuple(Tuple(R,G,B))``, list of colors format RGB
    """
    colors = np.random.randint(80, 255, size=(len(classes), 3))
    colors = tuple(map(tuple, colors))
    return colors


def assign_colors(classes):
    """
    Assigns colors to specific classes
    :param classes: ``Dict[class_id, "name of class"]``, dictionary of class names
    :return: ``Tuple(Tuple(R,G,B))``, list of colors format RGB
    """
    colors = np.random.randint(80, 255, size=(len(classes), 3), dtype='int32')
    # bio
    colors[1] = [204, 6, 5]  # person
    colors[16] = [118, 255, 122]  # bird
    colors[17] = [229, 81, 55]  # cat
    colors[18] = [219, 215, 210]  # dog
    # vehicles
    colors[2] = [0, 149, 182]  # bicycle
    colors[3] = [127, 255, 212]  # car
    colors[4] = [205, 164, 222]  # motorcycle
    colors[5] = [249, 132, 229]  # airplane
    colors[6] = [248, 243, 43]  # bus
    colors[7] = [100, 149, 237]  # train
    colors[8] = [222, 76, 138]  # truck
    # other
    colors[10] = [237, 118, 14]  # traffic light

    colors = tuple(map(tuple, colors))
    return colors


def apply_mask(image, mask, color, threshold=0.5, alpha=0.5):
    """
    Applying mask to image and thresholding
    :param image: ``Numpy array [H, W, 3]``
    :param mask: ``Numpy array[H, W]``
    :param color: ``Tuple(R,G,B)``, list of colors format RGB
    :param threshold: soft masks
    :param alpha: pixel overlay opacity
    :return:
    """
    for c in range(3):
        image[..., c] = np.where(mask >= threshold,
                                 image[..., c] * (1 - alpha) + alpha * color[c], image[..., c])

    return image


def draw_contours(image, mask, color):
    """
    Draws contours around mask
    :param image: ``Numpy array [H, W, 3]``
    :param mask: ``Numpy array[H, W]``
    :param color: ``Tuple(R,G,B)``, list of colors format RGB
    :return: ``list`` list of contours
    """
    color_contour = tuple(map(int, color))
    _, rough_mask = cv2.threshold(mask.squeeze(0) * 255, thresh=cfg.MASK_ROUGH_THRESHOLD,
                                  maxval=cfg.MASK_ROUGH_MAXVAL, type=cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(rough_mask.astype(np.uint8),
                                           mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(image, contours, contourIdx=0, color=color_contour,
                     thickness=cfg.MASK_CONTOUR_THICKNESS)

    return contours


def get_center(mask):
    """
    Finds center of object
    :param mask: ``Numpy array[H, W]``
    :return: if contours do not intersect return (x, y) coords of center object else (None, None)
    """
    _, rough_mask = cv2.threshold(mask.squeeze(0) * 255, thresh=cfg.MASK_ROUGH_THRESHOLD,
                                  maxval=cfg.MASK_ROUGH_MAXVAL, type=cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(rough_mask.astype(np.uint8),
                                           mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
    m = cv2.moments(contours[0])

    try:
        x_center = int(m["m10"] / m["m00"])
        y_center = int(m["m01"] / m["m00"])
    except ZeroDivisionError:
        return None, None
    else:
        return x_center, y_center


def draw_center_object(image, x_center, y_center):
    """
    Draws circle in the center of object
    :param x_center: x coord of center
    :param y_center: y coord of center
    :param image: ``Numpy array [H, W, 3]``
    """
    cv2.circle(image, (x_center, y_center), radius=5, color=(0, 0, 0), thickness=-1)
    cv2.putText(image, "center", (x_center - 20, y_center - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


def display_objects(images, predictions, cls_names, colors, display_boxes,
                    display_masks, display_caption, display_contours, remove_background=False):
    """
    Display objects on images
    :param images: ``List[[Tensor]]``, list of images (B,G,R)
    :param predictions:
    ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with values between
          ``0`` and ``H`` and ``0`` and ``W``
        - labels (``Int64Tensor[N]``): the predicted labels for each image
        - scores (``Tensor[N]``): the scores or each prediction
        optional
        - masks (``UInt8Tensor[N, 1, H, W]``): the predicted masks for each instance, in ``0-1`` range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (``mask >= 0.5``)
    :param cls_names: ``Dict[class_id, "name of class"]``, dictionary of class names
    :param colors: ``Tuple(Tuple(R,G,B))``, list of colors format RGB
    :param display_boxes: if True: displays bounding boxes on images
    :param display_masks: if True: displays masks on images
    :param display_caption: if True: displays caption on images
    :param display_contours: if True - displays contours around mask on image
    :param remove_background: if True - replace background from image
    :return ``List[[numpy_array]]``, list of images
    """
    image_list = []

    for k, prediction in enumerate(predictions):
        image = Image.fromarray(utils.reverse_normalization(images[k]))
        boxes = prediction['boxes'].cpu()
        masks = prediction['masks'].cpu().numpy() if 'masks' in prediction else None
        labels = prediction['labels'].cpu().detach().numpy()
        scores = prediction['scores'].cpu().detach().numpy()
        draw = ImageDraw.Draw(image)
        num_boxes = boxes.shape[0]

        for i in range(num_boxes):
            cls_id = labels[i]
            x1, y1, x2, y2 = boxes[i]

            if display_boxes:
                draw.rectangle(xy=((x1, y1), (x2, y2)), outline=colors[cls_id], width=cfg.THICKNESS_BBOX)

            if display_caption:
                class_name = cls_names[cls_id]
                score = scores[i]
                caption = '{} {}%'.format(class_name, round(score * 100, 1))
                # FIXME: exception doesn't work
                try:
                    font = ImageFont.truetype(cfg.PATH_TO_FONT, cfg.FONT_SIZE)
                except IOError:
                    logger.exception('Font error')
                    font = ImageFont.load_default()

                text_size = draw.textsize(caption, font)
                draw.rectangle(xy=((x1, y1 - text_size[1] - cfg.HEIGHT_TEXT_BBOX),
                                   (x1 + text_size[0] + cfg.WIDTH_TEXT_BBOX, y1)),
                               fill=colors[cls_id])
                draw.text((x1 + 2, y1 - text_size[1]), caption, font=font, fill=(0, 0, 0))

        if 1 in {display_masks, display_contours, cfg.DISPLAY_CENTER_OBJECT}:
            if remove_background:
                height, width, channels = image.shape
                image = np.zeros((height, width, channels), np.uint8)
            else:
                image = np.array(image, dtype=np.uint8)

            num_masks = masks.shape[0]
            for i in range(num_masks):
                cls_id = labels[i]
                mask = masks[i, ...]
                if display_masks:
                    apply_mask(image, mask, colors[cls_id], threshold=cfg.MASK_THRESHOLD, alpha=cfg.MASK_ALPHA)

                if display_contours:
                    draw_contours(image, mask, colors[cls_id])

                if cfg.DISPLAY_CENTER_OBJECT:
                    x_center, y_center = get_center(mask)
                    if (x_center and y_center) is not None:
                        draw_center_object(image, x_center, y_center)

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        image_list.append(image)
    return image_list
