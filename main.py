import argparse
import torch
import os
import models
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
    parser.add_argument('--flip', dest='flip', help='flip video. Warning: expensive operation',
                        action='store_true')
    parser.add_argument('--use_gpu', dest='use_gpu',
                        help='whether use GPU, if the GPU is unavailable then the CPU will be used',
                        action='store_true')
    parser.print_help()
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:{}'.format(args.__dict__))

    if not os.path.exists(args.outdir):
        os.makedirs('output')
        args.outdir = 'output'

    if args.images is args.video:
        raise RuntimeError('path to images and videos not specified')

    if torch.cuda.is_available() and not args.use_gpu:
        print('You have a GPU device, so you should probably run with --use_gpu')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:{}'.format(device))

    model = models.get_model_faster_rcnn()

    model.to(device)
    model.eval()
    detector = Detector(model, device)

    if args.images:
        detector.detect_on_images(args.images, args.outdir, threshold=cfg.THRESHOLD)
    elif args.video:
        detector.detect_on_video(args.video, args.outdir, threshold=cfg.THRESHOLD, flip=args.flip)
    else:
        raise RuntimeError('Something went wrong...')
