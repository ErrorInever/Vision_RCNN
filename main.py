import argparse
import torch
import os
import models
import logging
import time
from detection.detector import Detector
from config.cfg import cfg


def parse_args():
    parser = argparse.ArgumentParser(description='Faster-RCNN')
    parser.add_argument('--images', dest='images', help='Path to directory where images stored',
                        default=None, type=str)
    parser.add_argument('--video', dest='video', help='path to directory where video stored',
                        default=None, type=str)
    parser.add_argument('--outdir', dest='outdir',
                        help='directory to save results, default save to /output',
                        default='output', type=str)
    parser.add_argument('--flip_video', dest='flip', help='flip video. Warning: expensive operation',
                        action='store_true')
    parser.add_argument('--use_gpu', dest='use_gpu',
                        help='whether use GPU, if the GPU is unavailable then the CPU will be used',
                        action='store_true')
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    logger = logging.getLogger()
    c_handler = logging.FileHandler('logs/inference_logs.log', mode='w')
    #f_handler = logging.FileHandler('logs/error_logs.log', mode='w')

    c_handler.setLevel(logging.INFO)
    #f_handler.setLevel(logging.ERROR)
    c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
    #f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    c_handler.setFormatter(c_format)
    #f_handler.setFormatter(f_format)

    logger.addHandler(c_handler)
    #logger.addHandler(f_handler)

    logger.setLevel(logging.INFO)

    args = parse_args()

    logger.info('Mask RCNN start {}'.format(time.ctime()))
    logger.info('Config params:{}'.format(cfg.__dict__))
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

    logger.info('Using device %s', device)

    logger.info('Set up model')
    model = models.get_model_mask_rcnn()
    model.eval()
    model.to(device)
    detector = Detector(model, device)

    if args.images:
        detector.detect_on_images(args.images, args.outdir)
    elif args.video:
        detector.detect_on_video(args.video, args.outdir, flip=args.flip)
    else:
        raise RuntimeError('Something went wrong...')
