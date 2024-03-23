import torch

""" Import the configuration file """
import config as cfg
import timm


class ImageEncoder(torch.nn.Module):
    def __init__(self, model_name: str = cfg.model_name, pretrained: str = cfg.pretrained, trainable: str = cfg.trainable) -> None:
        """
        Constructor for the image encoder:
        - Accepts the model name, pretrained and trainable parameters.
        - The model name is the name of the model used for image encoder.
        - The pretrained is a boolean indicating whether to use the pretrained weights for the image encoder.
        - The trainable is a boolean indicating whether to train the image encoder.
        - The model name set in the config.py file will be used to load the model from timm, so make sure the model name is correct.
        - If the pretrained parameter is set to True, the pretrained weights will be downloaded and then used, else the model will be trained from scratch.
        - If the trainable parameter is set to True, the model will be trained, else the model will be frozen.
        - The timm library is used to load the model and the global pooling is set to "avg" to get the image embeddings.
        - If needed, the image will be resized to fit the input size of the model.

        Args:
        - model_name: The name of the model used for image encoder. Default is the model name from the config file.
        - pretrained: A boolean indicating whether to use the pretrained weights for the image encoder. Default is the pretrained from the config file.
        - trainable: A boolean indicating whether to train the image encoder. Default is the trainable from the config file.

        Example:
        >>> image_encoder = ImageEncoder()
        >>> image_encoder = ImageEncoder(model_name="resnet50", pretrained=True, trainable=False)
        >>> image = torch.randn(num_images, 3, cfg.image_size, cfg.image_size)
        >>> output = image_encoder(image)
        >>> output.shape
        torch.Size([num_images, cfg.image_embedding_dimension])
        >>> output[0]
        tensor([[ 0.1234, -0.5678,  0.9876, ...]])
        """
        super(ImageEncoder, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0, global_pool="avg")

        for param in self.model.parameters():
            param.requires_grad = trainable

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the image encoder:
        - Accepts the inputs tensor which is the image tensor.
        - The image tensor is passed through the model to get the image embeddings.
        - The image embeddings are returned. The embedding size depends on the model used.

        Args:
        - inputs: The image tensor which is passed through the model.

        Returns:
        - The image embeddings which are the output of the model.

        Example:
        >>> image_encoder = ImageEncoder()
        >>> image = torch.randn(num_images, 3, cfg.image_size, cfg.image_size)
        >>> output = image_encoder(image)
        >>> output.shape
        torch.Size([num_images, cfg.image_embedding_dimension]) # Projection will be done later
        >>> output[0]
        tensor([[ 0.1234, -0.5678,  0.9876, ...]])
        """
        return self.model(inputs)