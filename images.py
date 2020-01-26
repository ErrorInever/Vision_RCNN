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
        self.img_names = [n for n in os.listdir(img_path) if n.endswith(('jpg', 'jpeg', 'png'))]

    def __getitem__(self, idx):
        img = Image.open(os.path.join(self.img_path, self.img_names[idx])).convert('RGB')
        return self.img_to_tensor(img)

    def __len__(self):
        return len(self.img_names)

    def __str__(self):
        return str(self.img_names[:5])

    @property
    def img_to_tensor(self):
        """Convert an image to a tensor"""
        return transforms.Compose([transforms.ToTensor()])
