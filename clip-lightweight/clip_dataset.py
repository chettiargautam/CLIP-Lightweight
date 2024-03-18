import os, cv2, torch
import albumentations as A

"""Importing the configuration file for the dataset"""
import config as cfg


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, captions_path, tokenizer, max_length, transform=None):
        """
        Constructor for the dataset:
        - image_paths: list of paths to the images
        - captions: list of captions for the images
        """