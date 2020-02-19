from easydict import EasyDict as edict

__C = edict()
# for consumers
cfg = __C

# params
__C.BATCH_SIZE = 10
__C.NUM_WORKERS = 4
__C.THRESHOLD = 0.7

# bbox
__C.THICKNESS_BBOX = 3
__C.HEIGHT_TEXT_BBOX = 1
__C.WIDTH_TEXT_BBOX = 4

# path to font
__C.PATH_TO_FONT = None
# font size
__C.FONT_SIZE = 10
__C.FONT_COLOR = (0, 0, 0)

