from datetime import datetime
import cv2
from config.cfg import cfg


def show_info(frame):
    text = "{data}\nTHRESHOLD= {thresh}\nBATCH_SIZE= {b_size}" \
           "\nNUM_WORKERS= {num_works}\n".format(
            data=datetime.now().date().strftime('%d-%m-%Y'), thresh=cfg.THRESHOLD, b_size=cfg.BATCH_SIZE,
            num_works=cfg.NUM_WORKERS)
    cv2.putText(frame, text, (1, 1), fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1, color=(0, 0, 0), thickness=1)
    pass
