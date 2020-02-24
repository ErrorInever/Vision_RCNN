import argparse
import torch
import os
import models
import logging
import time
from detection.detector import Detector
from config.cfg import cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Vision-RCNN')
    parser.add_argument('--images', dest='images', help='Path to directory where images stored', default=None, type=str)
    parser.add_argument('--video', dest='video', help='Path to directory where video stored', default=None, type=str)
    parser.add_argument('--outdir', dest='outdir', help='Directory to save results', default='output', type=str)
    parser.add_argument('--show_boxes', dest='show_boxes', help='Display bounding boxes', action='store_true')
    parser.add_argument('--show_masks', dest='show_masks', help='Display masks', action='store_true')
    parser.add_argument('--show_caption', dest='show_caption', help='Display caption', action='store_true')
    parser.add_argument('--show_contours', dest='show_contours', help='Display contour mask', action='store_true')
    parser.add_argument('--flip_video', dest='flip', help='Flip video', action='store_true')
    parser.add_argument('--use_gpu', dest='use_gpu', help='Whether use GPU', action='store_true')
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    logger = logging.getLogger()

    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(cfg.PATH_TO_LOG_FILE, mode='w', encoding='utf-8')

    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.ERROR)
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    logger.setLevel(logging.DEBUG)

    args = parse_args()

    logger.info('Mask RCNN start {}'.format(time.ctime()))
    logger.info('Called with args: {}'.format(args.__dict__))

    if not os.path.exists(args.outdir):
        os.makedirs('output')
        args.outdir = 'output'

    if args.images is None and args.video is None:
        logger.error('Path to image and videos not specified: img_path=%s, video_path=%s', args.images, args.video)
        raise IOError

    if torch.cuda.is_available() and not args.use_gpu:
        logger.info('You have a GPU device so you should probably run with --use_gpu')
        device = torch.device('cpu')
    elif torch.cuda.is_available() and args.use_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cpu')

    if args.show_boxes:
        cfg.DISPLAY_BOUNDING_BOXES = True
    if args.show_masks:
        cfg.DISPLAY_MASKS = True
    if args.show_caption:
        cfg.DISPLAY_CAPTION = True
    if args.show_contours:
        cfg.DISPLAY_CONTOURS = True
    if 1 not in {cfg.DISPLAY_BOUNDING_BOXES, cfg.DISPLAY_MASKS, cfg.DISPLAY_CAPTION, cfg.DISPLAY_CONTOURS}:
        logger.error('Nothing shows: show_boxes=%s, show_masks=%s, show_caption=%s, show_contours=%s',
                     cfg.DISPLAY_BOUNDING_BOXES, cfg.DISPLAY_MASKS, cfg.DISPLAY_CAPTION, cfg.DISPLAY_CONTOURS)
        raise IOError

    logger.info('Config params:{}'.format(cfg.__dict__))
    logger.info('Using device %s', device)

    logger.info('Set up model')
    model = models.get_model_mask_rcnn()
    model.eval()
    model.to(device)
    detector = Detector(model, device)

    if args.images:
        detector.detect_on_images(args.images, args.outdir, cfg.DISPLAY_MASKS, cfg.DISPLAY_BOUNDING_BOXES,
                                  cfg.DISPLAY_CAPTION, cfg.DISPLAY_CONTOURS)
    elif args.video:
        detector.detect_on_video(args.video, args.outdir, cfg.DISPLAY_MASKS, cfg.DISPLAY_BOUNDING_BOXES,
                                 cfg.DISPLAY_CAPTION, cfg.DISPLAY_CONTOURS, flip=args.flip)
