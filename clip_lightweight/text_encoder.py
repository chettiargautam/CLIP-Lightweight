from transformers import DistilBertModel, DistilBertConfig
import torch


""" Import the configuration file """
import config as cfg


class TextEncoder(torch.nn.Module):
    def __init__(self, model_name: str = cfg.text_encoder_model, pretrained: str = cfg.pretrained, trainable: str = cfg.trainable) -> None:
        """
        Constructor for the text encoder:
        - Accepts the model name, pretrained and trainable parameters.
        - The model name is the name of the model used for text encoder.
        - The pretrained is a boolean indicating whether to use the pretrained weights for the text encoder.
        - The trainable is a boolean indicating whether to train the text encoder.
        - The model name set in the config.py file will be used to load the model from HuggingFace, so make sure the model name is correct.
        - If the pretrained parameter is set to True, the pretrained weights will be downloaded and then used, else the model will be trained from scratch.
        - If the trainable parameter is set to True, the model will be trained, else the model will be frozen.
        - We use the CLS token hidden representation as the sentence's embedding
        - The model is loaded from the HuggingFace transformers library.

        Args:
        - model_name: The name of the model used for text encoder. Default is the model name from the config file.
        - pretrained: Boolean indicating whether to use the pretrained weights. Default is the pretrained value from the config file.
        - trainable: Boolean indicating whether to train the text encoder. Default is the trainable value from the config file.

        Example:
        >>> text_encoder = TextEncoder()
        >>> text = torch.randn(num_sentences, max_length)
        >>> output = text_encoder(text)
        >>> output.shape
        torch.Size([num_sentences, cfg.text_embedding_dimension])
        >>> output[0]
        tensor([[ 0.1234, -0.5678,  0.9876, ...]])
        >>> output[0].shape
        torch.Size([1, cfg.text_embedding_dimension])
        """
        super(TextEncoder, self).__init__()

        if pretrained:
            self.model = DistilBertModel.from_pretrained(model_name)
        else:
            config = DistilBertConfig.from_pretrained(model_name)
            self.model = DistilBertModel(config)

        for param in self.model.parameters():
            param.requires_grad = trainable

        self.target_token_index = 0

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the text encoder:
        - Accepts the inputs tensor which is the tokenized text input.
        - The tokenized text input_ids is passed through the model to get the text embeddings.
        - The text embeddings are returned. The embedding size depends on the model used.
        - We use the CLS token hidden representation as the sentence's embedding

        Args:
        - input_ids: The tokenized text input tensor.
        - attention_mask: The attention mask tensor for the input.

        Returns:
        - The text embeddings which are the output of the model.

        Example:
        >>> text_encoder = TextEncoder()
        >>> text = torch.randn(num_sentences, max_length)
        >>> attention_mask = torch.ones(num_sentences, max_length)
        >>> output = text_encoder(text, attention_mask)
        >>> output.shape
        torch.Size([num_sentences, cfg.text_embedding_dimension]) # Projection will be done later
        >>> output[0]
        tensor([[ 0.1234, -0.5678,  0.9876, ...]])
        >>> output[0].shape
        torch.Size([1, cfg.text_embedding_dimension])
        """
        output = self.model(input_ids, attention_mask)
        embeddings = output.last_hidden_state[:, self.target_token_index, :]

        return embeddings