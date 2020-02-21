import utils
import numpy as np
from config.cfg import cfg
from PIL import Image, ImageDraw, ImageFont


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
    colors = np.random.randint(80, 255, size=(len(classes), 3))
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
    :param image: numpy array
    :param mask:
    :param color:
    :param threshold:
    :param alpha:
    :return:
    """
    for c in range(3):
        image[..., c] = np.where(mask >= threshold,
                                 image[..., c] * (1 - alpha) + alpha * color[c], image[..., c])

    return image


def display_objects(images, predictions, cls_names, colors, display_boxes=True,
                    display_masks=True, display_caption=True, threshold=0.7):
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
    :param threshold: removes predictions < threshold
    :return ``List[[numpy_array]]``, list of images
    """
    predictions = utils.filter_prediction(predictions, threshold)
    image_list = []
    for k, prediction in enumerate(predictions):
        boxes = prediction['boxes'].cpu()
        labels = prediction['labels'].cpu().detach().numpy()
        scores = prediction['scores'].cpu().detach().numpy()
        masks = prediction['masks'].cpu().numpy() if 'masks' in prediction else None
        image = Image.fromarray(utils.reverse_normalization(images[k]))
        draw = ImageDraw.Draw(image)
        masked_image = np.array(image)

        num_objects = boxes.shape[0]
        for i in range(num_objects):
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
                mask = masks[i, ...]
                masked_image = apply_mask(masked_image, mask, colors[cls_id], threshold=0.5, alpha=0.5)

        image_list.append(image)

    return image_list

