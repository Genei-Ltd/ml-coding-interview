from torch.utils.data import Dataset
import torch
from model import Config
from model import GPT

# Tokenizer
class CharTokenizer:
    def __init__(self, text):
        """
        Take in the full pretraining text, compute vocab size.
        """

    def encode(self, xs):
        """
        Input sequence of chars.
        Output torch LONG tensor of indicies.
        """

    def decode(self, xs):
        """
        Input torch long tensor of indicies.
        Output character string.
        """

# Dataset
class CharDataset(Dataset):
    def __init__(self, fpath, block_size):
        """
        Read in a file of pretraining text.
        Create a tokenizer from corpus.
        """

    def __getitem__(self, idx):
        """
        For each idx return a block_size block of tensors.
        Specificially return dict with keys.
        - x: input indicies (torch long tensor)
        - y: next token target indicies (torch long tensor)
        """

    def __len__(self):
        """
        Length of dataset.
        """

