import torch

""" Import the configuration file """
import config as cfg


class ProjectionHead(torch.nn.Module):
    def __init__(self, embedding_dimension: int, projection_dimension: int = cfg.projection_dimension, dropout: int = cfg.dropout) -> None:
        """
        Constructor for the projection head:
        - Accepts the embedding dimension, projection dimension and dropout parameters.
        - The embedding dimension is the dimension of the input embeddings.
        - The projection dimension is the dimension of the projected embeddings.
        - The dropout is the dropout used for the projection head.
        - The projection head consists of multiple linear layers with ReLU activation and dropout.
        - The final layer projects the embeddings to the projection dimension.

        Args:
        - embedding_dimension: The dimension of the input embeddings.
        - projection_dimension: The dimension of the projected embeddings. Default is the projection dimension from the config file.
        - dropout: The dropout used for the projection head. Default is the dropout from the config file.

        Example:
        >>> projection_head = ProjectionHead(embedding_dimension=768) # For text embeddings
        >>> projection_head = ProjectionHead(embedding_dimension=2048) # For image embeddings
        >>> embeddings = torch.randn(num_samples, embedding_dimension)
        >>> projected_embeddings = projection_head(embeddings)
        >>> projected_embeddings.shape
        torch.Size([num_samples, projection_dimension])
        """
        super(ProjectionHead, self).__init__()
        self.projection = torch.nn.Sequential(
            torch.nn.Linear(embedding_dimension, projection_dimension),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(projection_dimension, projection_dimension),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(projection_dimension, projection_dimension)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the projection head:
        - Accepts the inputs tensor which is the input embeddings.
        - The input embeddings are passed through the projection head to get the projected embeddings.

        Args:
        - inputs: The input embeddings which are passed through the projection head.

        Returns:
        - The projected embeddings which are the output of the projection head.

        Example:
        >>> projection_head = ProjectionHead(embedding_dimension=768)
        >>> embeddings = torch.randn(num_samples, 768)
        >>> projected_embeddings = projection_head(embeddings)
        >>> projected_embeddings.shape
        torch.Size([num_samples, cfg.projection_dimension])
        """
        return self.projection(inputs)