import torchvision


def get_model_faster_rcnn():
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True)


def get_model_mask_rcnn():
    return torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, progress=True)