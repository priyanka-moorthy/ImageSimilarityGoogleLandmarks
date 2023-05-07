__all__ = ["FolderDataset"]

import torch
from PIL import Image
import os
from tqdm import tqdm
from torch.utils.data import Dataset


class FolderDataset(Dataset):
    """
    Creates a PyTorch dataset from folder, returning two tensor images.
    Args:
    main_dir : directory where images are stored.
    transform (optional) : torchvision transforms to be applied while making dataset
    """

    def __init__(self, main_dir, transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.all_imgs_dir = self.index_dir(main_dir)

    def index_dir(self, directory):
        files = []
        for subdir, dirs, files in os.walk(directory):
            for file in files:
                files.append(os.path.join(subdir, file))
        return files
       


    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")

        if self.transform is not None:
            tensor_image = self.transform(image)

        return tensor_image, tensor_image