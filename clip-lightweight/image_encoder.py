import torch

""" Import the configuration file """
import config as cfg


class ImageEncoder(torch.nn.Module):
    def __init__(self, model_name: str = cfg.model_name, pretrained: str = cfg.pretrained, trainable: str = cfg.trainable) -> None:
        """
        Constructor for the image encoder:
        - Accepts the model name, pretrained and trainable parameters.
        - The model name is the name of the model used for image encoder.
        - The pretrained is a boolean indicating whether to use the pretrained weights for the image encoder.
        - The trainable is a boolean indicating whether to train the image encoder.
        - Currently can pick between 'resnet50', 'vgg16' and 'vit_base_patch16_224'.
        - If the pretrained parameter is set to True, the pretrained weights will be downloaded and then used, else the model will be trained from scratch.
        - If the trainable parameter is set to True, the model will be trained, else the model will be frozen.
        - If needed, the image will be resized to fit the input size of the model.
        """
        super(ImageEncoder, self).__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.trainable = trainable

        if self.model_name == 'resnet50':
            from torchvision.models import resnet50
            self.model = resnet50(pretrained=self.pretrained)
            self.model.fc = torch.nn.Identity()
            self.image_size = 224
        elif self.model_name == 'vgg16':
            from torchvision.models import vgg16
            self.model = vgg16(pretrained=self.pretrained)
            self.model.classifier[-1] = torch.nn.Identity()
            self.image_size = 224
        elif self.model_name == 'vit_base_patch16_224':
            from timm.models import vision_transformer
            self.model = vision_transformer(pretrained=self.pretrained)
            self.model.head = torch.nn.Identity()
            self.image_size = 224