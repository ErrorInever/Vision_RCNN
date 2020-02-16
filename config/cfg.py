from easydict import EasyDict as edict

__C = edict()
cfg = __C

# main
__C.BATCH_SIZE = 10
__C.NUM_WORKERS = 4
__C.THRESHOLD = 0.7

# bbox
__C.THICKNESS_BBOX = 3

# text size
__C.TEXT_SIZE = 1
