from dotenv import load_dotenv
import os
import argparse

load_dotenv()

"""
INSTRUCTIONS:
This file is used to load the environment variables from the .env file. Run this file before running any other file in the cmd line.
- The .env file should be in the same directory as this file, set your credentials to the KAGGLE_USERNAME and KAGGLE_KEY variables accordingly.
>>> cd CLIP-Lightweight/
>>> pwd
/Users/username/CLIP-Lightweight
>>> python kaggle_init.py
"""

os.environ['KAGGLE_USERNAME'] = os.getenv("KAGGLE_USERNAME")
os.environ['KAGGLE_KEY'] = os.getenv("KAGGLE_KEY")


def dataset_downloader(dataset_size: str = '8k', root_directory: str = 'data/') -> None:
    """
    Dataset Downloader:
    - Takes the input either 8k (default) or 30k and downloads the respective dataset, as well as the directory to install the data.
    - Downloads the kaggle dataset using the kaggle API and stores it in the specified directory (set to CLIP-Lightweight/data by default).
    - Runs the command line instructions using os as this is a python file. This isn't implemented as a colab notebook.
    -- !kaggle datasets download -d adityajn105/flickr8k
    -- !unzip flickr8k.zip
    -- or
    -- !kaggle datasets download -d hsankesara/flickr-image-dataset
    -- !unzip flickr-image-dataset.zip
    Args:
    - dataset_size: str: Takes the input either 8k or 30k. Set to 8k by default.
    - root_directory: str: The directory to install the data. Set to CLIP-Lightweight/data by default.
    Returns:
    - None

    Example:
    >>> cd CLIP-Lightweight/
    >>> pwd
    /Users/username/CLIP-Lightweight
    >>> python kaggle_init.py
    """
    if dataset_size == '8k':
        os.system(f'kaggle datasets download -d adityajn105/flickr8k -p {root_directory}')
        os.system(f'unzip {root_directory}/flickr8k.zip -d {root_directory}')
    elif dataset_size == '30k':
        os.system(f'kaggle datasets download -d hsankesara/flickr-image-dataset -p {root_directory}')
        os.system(f'unzip {root_directory}/flickr-image-dataset.zip -d {root_directory}')


def main() -> None:
    """
    Main Function:
    - Uses argparse to parse the command line arguments. Looks to receive the dataset size and the root directory, otherwise defaults to 8k and CLIP-Lightweight/data.
    Args:
    - None
    Returns:
    - None
    """
    parser = argparse.ArgumentParser(description='Dataset Downloader')
    parser.add_argument('--dataset_size', type=str, default='8k', help='Dataset Size: 8k or 30k')
    parser.add_argument('--root_directory', type=str, default='data/', help='Root Directory: CLIP-Lightweight/data')
    args = parser.parse_args()

    if args.dataset_size == "30k":
        print(f"--> Downloading the {args.dataset_size} dataset in the {args.root_directory} directory")
        dataset_downloader(dataset_size=args.dataset_size, root_directory=args.root_directory)
    else:
        print(f"--> Downloading the 8k dataset in the {args.root_directory} directory")
        dataset_downloader(dataset_size=args.dataset_size, root_directory=args.root_directory)

    print("--> Dataset Downloaded Successfully!")


if __name__ == "__main__":
    main()