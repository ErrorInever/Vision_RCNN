import os
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image


class Images(Dataset):
    """
    Container for images
    """
    def __init__(self, img_path):
        self.img_path = img_path
        self.img_names = [n for n in os.listdir(img_path)]

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_path, self.img_names[idx])).convert('RGB')
        return self.img_to_tensor(img)

    def __len__(self):
        return len(self.img_names)

    @property
    def img_to_tensor(self):
        """Convert an image to a tensor"""
        return transforms.Compose([transforms.ToTensor()])

