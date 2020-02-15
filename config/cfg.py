from easydict import EasyDict as edict

__C = edict()
cfg = __C

# main
__C.BATCH_SIZE = 10
__C.NUM_WORKERS = 4
__C.NUM_CLASSES = 81
__C.THRESHOLD = 0.7

# interface
__C.INTERFACE = edict()

__C.INTERFACE.IMG_PATH = None
__C.INTERFACE.VIDEO_PATH = None

__C.BORDER_LINE_WIDTH = 1
