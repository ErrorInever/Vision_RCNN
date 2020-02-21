from easydict import EasyDict as edict

__C = edict()
# for consumers
cfg = __C

# data params
__C.BATCH_SIZE = 10
__C.NUM_WORKERS = 4

# thresholds
__C.SCORE_THRESHOLD = 0.7
__C.MASK_THRESHOLD = 0.5
__C.MASK_ALPHA = 0.5

# bounding boxes display params
__C.THICKNESS_BBOX = 3
__C.HEIGHT_TEXT_BBOX = 1
__C.WIDTH_TEXT_BBOX = 4

# fonts params
__C.PATH_TO_FONT = 'FasterRCNN_implementation/config/fonts/Ubuntu-B.ttf'
__C.FONT_SIZE = 14
__C.FONT_COLOR = (0, 0, 0)

