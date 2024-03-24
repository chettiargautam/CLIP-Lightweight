import torch, typing
from clip_lightweight.image_encoder import ImageEncoder
from clip_lightweight.text_encoder import TextEncoder
from clip_lightweight.projection import ProjectionHead

""" Import the configuration file """
import config as cfg


class CLIP(torch.nn.Module):
    def __init__(
            self, 
            temperature: float = cfg.temperature, 
            image_embedding_dimension: int = cfg.image_embedding_dimension, 
            text_embedding_dimension: int = cfg.text_embedding_dimension
        ) -> None:
        """
        Constructor for the CLIP model:
        - Accepts the temperature, image embedding dimension and text embedding dimension parameters.
        - The temperature is the temperature used for the contrastive loss.
        - The image embedding dimension is the dimension of the image embeddings.
        - The text embedding dimension is the dimension of the text embeddings.
        - The CLIP model consists of the image encoder, text encoder and projection head.
        - The image encoder is used to get the image embeddings.
        - The text encoder is used to get the text embeddings.
        - The projection head is used to project the image and text embeddings.

        Args:
        - temperature: The temperature used for the contrastive loss. Default is the temperature from the config file.
        - image_embedding_dimension: The dimension of the image embeddings. Default is the image embedding dimension from the config file.
        - text_embedding_dimension: The dimension of the text embeddings. Default is the text embedding dimension from the config file.

        Example:
        >>> clip = CLIP()
        >>> num_samples = num_images = num_texts
        >>> images = torch.randn(num_samples, 3, cfg.image_size, cfg.image_size) # (num_images, 3, cfg.image_size, cfg.image_size)
        >>> texts = torch.randn(num_samples, cfg.max_length) # (num_texts, cfg.max_length)
        >>> image_embeddings = clip.image_encoder(images)
        >>> text_embeddings = clip.text_encoder(texts)
        >>> image_embeddings.shape, text_embeddings.shape
        (torch.Size([num_samples, cfg.image_embedding_dimension]), torch.Size([num_samples, cfg.text_embedding_dimension]))
        >>> projected_image_embeddings = clip.image_projection(image_embeddings)
        >>> projected_text_embeddings = clip.text_projection(text_embeddings)
        >>> projected_image_embeddings.shape, projected_text_embeddings.shape
        (torch.Size([num_samples, cfg.projection_dimension]), torch.Size([num_samples, cfg.projection_dimension]))
        """
        super(CLIP, self).__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.image_projection = ProjectionHead(image_embedding_dimension)
        self.text_projection = ProjectionHead(text_embedding_dimension)
        self.temperature = temperature

    def forward(self, batch: dict) -> typing.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CLIP model:
        - Accepts the batch dictionary which contains the images and texts.
        - The images are passed through the image encoder to get the image embeddings.
        - The texts are passed through the text encoder to get the text embeddings.
        - The image and text embeddings are projected using the projection head.
        - The projected image and text embeddings are returned.
        - The logits are calculated using the projected image and text embeddings.
        - The similarity matrices are calculated using the image and text embeddings.
        - The loss is calculated using the logits and the similarity matrices.

        Args:
        - batch: A dictionary containing the images and texts.

        Returns:
        - The loss value.
        - The projected image embeddings.
        - The projected text embeddings.

        Example:
        >>> clip = CLIP()
        >>> num_samples = num_images = num_texts
        >>> images = torch.randn(num_samples, cfg.image_size, cfg.image_size, 3) # (num_images, cfg.image_size, cfg.image_size, 3)
        >>> texts = torch.randn(num_samples, cfg.max_length) # (num_texts, cfg.max_length)
        >>> inputs = {"images": images, "texts": texts}
        >>> _, image_embeddings, text_embeddings = clip(inputs)
        >>> image_embeddings.shape, text_embeddings.shape
        (torch.Size([num_samples, cfg.projection_dimension]), torch.Size([num_samples, cfg.projection_dimension]))
        """
        image_embeddings = self.image_encoder(batch["images"])
        text_embeddings = self.text_encoder(batch["input_ids"], batch["attention_mask"])

        image_embeddings = self.image_projection(image_embeddings)
        text_embeddings = self.text_projection(text_embeddings)

        logits = torch.matmul(image_embeddings, text_embeddings.T) / self.temperature
        images_similarity = torch.matmul(image_embeddings, image_embeddings.T) / self.temperature
        texts_similarity = torch.matmul(text_embeddings, text_embeddings.T) / self.temperature

        loss = (2 * torch.logsumexp(logits, dim=-1) - torch.logsumexp(images_similarity, dim=-1) - torch.logsumexp(texts_similarity, dim=-1)).mean()

        return loss, image_embeddings, text_embeddings