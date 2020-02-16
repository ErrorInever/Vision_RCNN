import torchvision


def get_model_faster_rcnn():
    return torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, progress=True)
