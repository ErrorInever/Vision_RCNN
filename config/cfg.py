from easydict import EasyDict as edict

__C = edict()
# for consumers
cfg = __C

# displaying
__C.DISPLAY_BOUNDING_BOXES = False
__C.DISPLAY_MASKS = False
__C.DISPLAY_CAPTION = False
__C.DISPLAY_CONTOURS = False
__C.DISPLAY_CENTER_OBJECT = False

# data params
__C.BATCH_SIZE = 10
__C.NUM_WORKERS = 4

# object threshold
__C.SCORE_THRESHOLD = 0.7

# mask
__C.MASK_THRESHOLD = 0.5
__C.MASK_ROUGH_THRESHOLD = 130
__C.MASK_ROUGH_MAXVAL = 255
__C.MASK_ALPHA = 0.5
__C.MASK_CONTOUR_THICKNESS = 2

# bounding boxes display params
__C.THICKNESS_BBOX = 3
__C.HEIGHT_TEXT_BBOX = 1
__C.WIDTH_TEXT_BBOX = 4

# fonts params
__C.PATH_TO_FONT = 'Vision_RCNN/config/fonts/Ubuntu-B.ttf'
__C.FONT_SIZE = 14
__C.FONT_COLOR = (0, 0, 0)
