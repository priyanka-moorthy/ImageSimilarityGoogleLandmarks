__all__ = ["FolderDataset"]

import torch
from PIL import ImageFile,Image
import os
from tqdm import tqdm
from torch.utils.data import Dataset
import torchvision.transforms as T
import config
ImageFile.LOAD_TRUNCATED_IMAGES = True


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
        img_files = [os.path.join(path, name) for path, subdirs, files in os.walk(directory) for name in files]
        return img_files

    def __len__(self):
        return len(self.all_imgs_dir)

    def __getitem__(self, idx):
        img_loc =  self.all_imgs_dir[idx]
        image = Image.open(img_loc).convert("RGB")
        transform_size = T.Resize((config.IMG_WIDTH,config.IMG_HEIGHT))
        resized_img = transform_size(image)
        if self.transform is not None:
            tensor_image = self.transform(resized_img)
        return tensor_image, tensor_image