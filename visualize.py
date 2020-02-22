import numpy as np
import cv2
from config.cfg import cfg
from PIL import Image, ImageDraw, ImageFont
from detection import utils


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


# TODO: border around mask, find center of object, video inference, draws overlap, iou, layers of net
def display_objects(images, predictions, cls_names, colors, display_boxes,
                    display_masks, display_caption, score_threshold):
    """
    Display objects on images
    :param images: ``List[[Tensor]]``, list of images
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
    :param score_threshold: removes predictions < threshold
    :return ``List[[numpy_array]]``, list of images
    """
    predictions = utils.filter_prediction(predictions, score_threshold)
    image_list = []

    for k, prediction in enumerate(predictions):
        boxes = prediction['boxes'].cpu()
        labels = prediction['labels'].cpu().detach().numpy()
        scores = prediction['scores'].cpu().detach().numpy()
        masks = prediction['masks'].cpu().numpy() if 'masks' in prediction else None

        image = Image.fromarray(utils.reverse_normalization(images[k]))

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
                    font = ImageFont.load_default()

                text_size = draw.textsize(caption, font)
                draw.rectangle(xy=((x1, y1 - text_size[1] - cfg.HEIGHT_TEXT_BBOX),
                                   (x1 + text_size[0] + cfg.WIDTH_TEXT_BBOX, y1)),
                               fill=colors[cls_id])
                draw.text((x1 + 2, y1 - text_size[1]), caption, font=font, fill=(0, 0, 0))

        if display_masks and (masks is not None):
            num_masks = masks.shape[0]
            image = np.array(image, dtype=np.uint8)
            for i in range(num_masks):
                mask = masks[i, ...]
                cls_id = labels[i]
                color_contour = tuple(map(int, colors[cls_id]))
                apply_mask(image, mask, colors[cls_id], threshold=cfg.MASK_THRESHOLD, alpha=cfg.MASK_ALPHA)
                # draw contours around mask
                _, rough_mask = cv2.threshold(mask.squeeze(0) * 255, thresh=cfg.MASK_ROUGH_THRESHOLD,
                                              maxval=cfg.MASK_ROUGH_MAXVAL, type=cv2.THRESH_BINARY)
                contours, hierarchy = cv2.findContours(rough_mask.astype(np.uint8),
                                                       mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(image, contours, contourIdx=0, color=color_contour,
                                 thickness=cfg.MASK_CONTOUR_THICKNESS)

        if not isinstance(image, np.ndarray):
            image = np.array(image)

        image_list.append(image)
    return image_list
