Project for own visualization of like-rcnn nets.

### Works on python 3.7 and lower

### Start
1. > clone this repo:  ``git clone https://github.com/ErrorInever/Vision_RCNN.git``
2. > install requirements: ``pip install -r requirements.txt`` 


## command line:

      optional arguments:
      -h, --help       show this help message and exit
      --images IMAGES  Path to directory where images stored
      --video VIDEO    Path to directory where video stored
      --outdir OUTDIR  Directory to save results
      --show_boxes     Display bounding boxes
      --show_masks     Display masks
      --show_caption   Display caption
      --show_contours  Display contour mask
      --flip_video     Flip video
      --use_gpu        Whether use GPU
      usage: main.py [-h] [--images IMAGES] [--video VIDEO] [--outdir OUTDIR]
                   [--show_boxes] [--show_masks] [--show_caption]
                   [--show_contours] [--flip_video] [--use_gpu]


## Example run: 

`python Vision_RCNN/main.py --video video/NAME.MP4 --show_boxes --show_mask --show_caption --show_contours --use_gpu`

`python Vision_RCNN/main.py --images image --show_boxes --show_mask --show_caption --show_contours --use_gpu`

### you can edit **config/cfg.py**
        
        If True - displaying bounding boxes
        __C.DISPLAY_BOUNDING_BOXES = False
        
        If True - displaying masks
        __C.DISPLAY_MASKS = False
        
        If True - displaying captions
        __C.DISPLAY_CAPTION = False
        
        If True - displaying contours aroung objects
        __C.DISPLAY_CONTOURS = False
        
        If True - displaying center of objects
        __C.DISPLAY_CENTER_OBJECT = False
        
        
![alt text](https://raw.githubusercontent.com/ErrorInever/Vision_RCNN/master/images/3-mALOBsyIY.jpg)
![alt text](https://raw.githubusercontent.com/ErrorInever/Vision_RCNN/master/images/ziVb5vyrfLU.jpg)
