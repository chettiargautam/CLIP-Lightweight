import torch

"""
Configurations for the project
--------------------------------
- All paths start from the root of the project.
- The image_path is the directory where the images are stored.
- The captions_path is the path to the file containing the captions.
- The batch_size is the number of samples in each batch.
- The num_workers is the number of workers used for loading the data.
- The lr is the learning rate for the optimizer.
- The weight_decay is the weight decay for the optimizer.
- The patience is the number of epochs with no improvement after which learning rate will be reduced.
- The factor is the factor by which the learning rate will be reduced.
- The epochs is the number of epochs to train the model.
- The device is the device used for training the model.
- The model_name is the name of the model used for image encoder.
- The image_embedding_dimension is the dimension of the image embedding.
- The text_encoder_model is the name of the model used for text encoder.
- The text_embedding_dimension is the dimension of the text embedding.
- The text_tokenizer is the name of the tokenizer used for text encoder.
- The max_length is the maximum length of the input sequence.

- The pretrained is a boolean indicating whether to use the pretrained weights for the image and text encoders.
- The trainable is a boolean indicating whether to train the image and text encoders.
- The temperature is the temperature used for the contrastive loss.

- The image_size is the size of the image used for training the model.

- The num_projection_layers is the number of layers used for the projection head.
- The projection_dimension is the dimension of the projection head.
- The dropout is the dropout used for the projection head.

(Taken from the original codebase by the authors of OpenAI-CLIP)
"""

debug = True
image_path = "data/images"
captions_path = "data/captions.txt"
batch_size = 8
num_workers = 0
lr = 1e-3
weight_decay = 1e-3
patience = 2
factor = 0.5
epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = 'resnet50'
image_embedding_dimension = 2048
text_encoder_model = "distilbert-base-uncased"
text_embedding_dimension = 768
text_tokenizer = "distilbert-base-uncased"
max_length = 200

pretrained = False 
trainable = False 
temperature = 1.0

image_size = 224

num_projection_layers = 1
projection_dimension = 256 
dropout = 0.1