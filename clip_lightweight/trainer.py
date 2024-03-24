import torch, warnings
from tqdm.auto import tqdm

from clip_lightweight.clip import CLIP
from clip_lightweight.clip_dataset import CLIPDataset, get_transforms

""" Import the configuration file """
import config as cfg


def train_model(
        num_epochs: int = cfg.epochs
        ) -> None:
    """
    Train the model:
    - Loads the data using the CLIPDataset class. This includes the images and the captions.
    - The CLIPDataset class returns the image and caption text (input_ids) for each item in the dataset.
    - The image is already in the correct tensor format and the caption text is tokenized and converted to a tensor.
    - The text which is tokenized must also get the attention mask and the padding mask.
    - Once the data is loaded, the model is trained using the contrastive loss.
    - The contrastive loss is calculated using the projected image and text embeddings.
    - The projected image and text embeddings are calculated using the projection head.
    - The model is trained for the specified number of epochs.
    - The trained model is saved after training with the name as "clip_model.pth".

    Parameters:
    - model: The CLIP model to be trained.
    - train_loader: The data loader for the training data.
    - val_loader: The data loader for the validation data.
    - criterion: The loss function used for training.
    - optimizer: The optimizer used for training.
    - num_epochs: The number of epochs for training. Default is the epochs from the config file.
    - device: The device used for training. Default is the device from the config file.

    Example:
    >>> model = CLIP()
    >>> train_loader = CLIPDataset(images_path="data/Images", captions_path="data/captions.txt", tokenizer=tokenizer, transforms=transforms)
    >>> val_loader = CLIPDataset(images_path="data/Images", captions_path="data/captions.txt", tokenizer=tokenizer, transforms=transforms)
    >>> criterion = torch.nn.CrossEntropyLoss()
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    >>> train_model(num_epochs=cfg.epochs)
    """
    model = CLIP()

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    train_loader = CLIPDataset(images_path=cfg.image_path, captions_path=cfg.captions_path, transforms=get_transforms)

    model.to(cfg.device)

    # Use tqdm in for loop to indicate epoch
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
    # for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc="Batches", leave=False):
            images = batch['image'].to(cfg.device)
            input_ids = batch['label']['input_ids'].to(cfg.device)
            attention_mask = batch['label']['attention_mask'].to(cfg.device)

            optimizer.zero_grad()
            loss, _, _ = model({
                "images": images,
                "input_ids": input_ids,
                "attention_mask": attention_mask
            })

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader)}")

    torch.save(model.state_dict(), 'clip_model.pth')


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    train_model()