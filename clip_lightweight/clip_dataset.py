import os, cv2, torch
import albumentations as A
from tqdm.auto import tqdm
from transformers import DistilBertTokenizer, DistilBertConfig

"""Importing the configuration file for the dataset"""
import config as cfg


"""Setting the image data type to 8-bit integer"""
IMAGE_DTYPE = torch.int8


class CLIPDataset(torch.utils.data.Dataset):
    def __init__(self, images_path: str, captions_path: str, transforms=None) -> None:
        """
        Constructor for the dataset:
        - Takes in the parameters which contain information on the images and captions.
        - Accepts the path of the images and the path of the captions.
        - Also accepts the tokenizer, max_length and the transform.
        - The images are read from the images_path and the captions are read from the captions_path.
        - The captions are tokenized using the tokenizer and the max_length.
        - The transforms are applied to the images during the __getitem__ function call.
        - Fully aware that one image has one or more captions, and this needs to be considered during training.

        Args:
        - images_path: Path to the images (directory). Default is 'data/Images/'.
        - captions_path: Path to the captions (filepath). Default is 'data/captions.txt'.
        - transforms: Transforms for the images. Default is None.

        Example:
        >>> dataset = CLIPDataset(images_path="data/Images", captions_path="data/captions.txt", tokenizer=tokenizer, transforms=transforms)
        >>> dataset.image_filenames
        ['image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg', ...]
        >>> dataset.captions
        ['A black cat is sitting on a white table.', 'A brown dog is running in the grass.', 'A white cat is sleeping on a brown couch.', ...]
        >>> dataset.encoded_captions
        {'input_ids': tensor([[  101,  1045,  1005,  2310,  1037,  2158,  1997,  1996,  1005,  2310, 1005, ...]), 
        'attention_mask': tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...]])}
        >>> dataset.encoded_captions['input_ids'].shape
        torch.Size([num_samples, cfg.max_length])
        >>> dataset.encoded_captions['attention_mask'].shape
        torch.Size([num_samples, cfg.max_length])
        """
        self.images_path = images_path
        self.captions_path = captions_path
        self.transforms = transforms

        self.image_filenames = []
        self.captions = []

        print("Loading the image file names and captions from the captions.txt file...")
        for line in tqdm(open(self.captions_path, 'r')):
            information = line.strip().split(',')
            image_filename, caption = information[0], information[1]
            self.image_filenames.append(image_filename.strip())
            self.captions.append(caption.strip())
        print("Done loading the image file names and captions from the captions.txt file...")

        self.image_filenames = self.image_filenames[1:]
        self.captions = self.captions[1:]

        self.encoded_captions = []

        tokenizer = DistilBertTokenizer.from_pretrained(cfg.text_encoder_model, do_lower_case=True, add_special_tokens=True, max_length=cfg.max_length, pad_to_max_length=True, return_tensors="pt")

        print("Encoding the captions into their respective embeddings...")
        self.encoded_captions = tokenizer(self.captions, max_length=cfg.max_length, padding="max_length", truncation=True, return_tensors="pt")
        print("Done encoding the captions into their respective embeddings...")

    def __len__(self) -> int:
        """
        Length of the dataset:
        - Returns the length of the dataset.
        """
        return len(self.captions)
    
    def __getitem__(self, index: int) -> dict:
        """
        Get item from the dataset:
        - Returns the item from the dataset based on the index.
        - This includes the image and the caption. Only one caption is returned for each image which depends on the index.
        - The image is read using OpenCV and then transformed using the transforms.
        - The caption is returned as a tensor.
        - Quantize the image to 8-bit and convert to a tensor.
        - Basically, this function returns the image and the caption for the index.
        - Avoids the issue of loading all the images and captions into memory at once.

        Args:
        - index: Index of the item to be returned from the dataset.

        Returns:
        - A dictionary containing the image and the caption. This is in the torch dataset format.
        - The image is a tensor and the caption is a dictionary containing the input_ids and the attention_mask.

        Example:
        >>> dataset = CLIPDataset(images_path="data/Images", captions_path="data/captions.txt", tokenizer=tokenizer, transforms=transforms)
        >>> dataset[0]
        {'image': tensor([[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        ...,
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000]]]), 
        'label': {
            'input_ids': tensor([[  101,  1045,  1005,  2310,  1037,  2158,  1997,  1996,  1005,  2310, 1005, ...]),
            'attention_mask': tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, ...]])
        }}
        >>> dataset[0]['image'].shape
        torch.Size([3, cfg.image_size, cfg.image_size])
        >>> dataset[0]['label']['input_ids'].shape
        torch.Size([1, cfg.max_length])
        >>> dataset[0]['label']['attention_mask'].shape
        torch.Size([1, cfg.max_length])
        """
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.images_path, image_filename)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transforms:
            image = self.transforms()(image=image)["image"]

        input_ids = self.encoded_captions["input_ids"][index]
        attention_mask = self.encoded_captions["attention_mask"][index]

        return {
            "image": torch.tensor(image),
            "label": {
                "input_ids": input_ids,
                "attention_mask": attention_mask
            }
        }
    

def get_transforms(mode: str = 'train') -> A.Compose:
    """
    Get the transforms for the images:
    - Returns the transforms for the images.
    - The transforms are different for training and validation.
    - The training transforms include random resized crop.
    - The validation transforms include resized crop.

    Args:
    - mode: Mode of the transforms. Default is 'train'.

    Returns:
    - The transforms for the images.

    Example:
    >>> get_transforms('train')
    Compose([
        RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.3333), interpolation=1),
    ], p=1.0, bbox_params=None, keypoint_params=None, additional_targets={})
    """
    if mode == 'train':
        return A.Compose([
            A.RandomResizedCrop(height=cfg.image_size, width=cfg.image_size, scale=(0.08, 1.0), ratio=(0.75, 1.3333), always_apply=True),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return A.Compose([
            A.Resize(height=cfg.image_size, width=cfg.image_size, always_apply=True),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])