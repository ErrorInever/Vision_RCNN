import argparse
import torch
import torchvision
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Faster-RCNN')
    parser.add_argument('--images', dest='images', help='Path to directory where images stored',
                        default=None, type=str)
    parser.add_argument('--video', dest='video', help='path to directory where video stored',
                        default=None, type=str)

    parser.add_argument('--outdir', dest='outdir',
                        help='directory to save results, default save to /output',
                        default='output', type=str)

    parser.add_argument('--use_gpu', dest='use_gpu',
                        help='whether use GPU, if the GPU is unavailable then the CPU will be used',
                        action='store_true')
    parser.print_help()
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    print('Called with args:{}'.format(args.__dict__))

    if not os.path.exists(args.outdir):
        os.makedirs('output')

    if args.images is args.video:
        raise RuntimeError('path to images and videos not specified')

    if torch.cuda.is_available() and not args.use_gpu:
        print('You have a GPU device, so you should probably run with --use_gpu')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Using device:{}'.format(device))

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True)

    if args.images:
        pass
    elif args.videos:
        pass
    else:
        raise RuntimeError('Something went wrong...')
