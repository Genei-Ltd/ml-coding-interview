from torch.utils.data import Dataset, DataLoader
import torch
import math
from model import Config
from model import GPT
from torch.optim import AdamW

# HYPERPARAMS
BLOCK_SIZE = 1024
LR=5e-4
EPOCHS=3
BATCH_SIZE=2
NGPUS = 4

# Tokenizer
class CharTokenizer:
    def __init__(self, text):
        vocab = set(text)
        self.vocab_size = len(vocab)
        self.stoi = {x:i for i,x in enumerate(vocab)}
        self.itos = {v:k for k,v in self.stoi.items()}

    def encode(self, xs):
        return torch.tensor([self.stoi[x] for x in xs], dtype=torch.long)

    def decode(self, xs):
        return ''.join([self.itos[x] for x in xs])

# Dataset
class CharDataset(Dataset):
    def __init__(self, fpath, block_size):
        self.fpath = fpath
        self.block_size = block_size

        with open(fpath, 'r') as f:
            self.text = f.read()

        self.tokenizer = CharTokenizer(self.text)

    def __getitem__(self, idx):
        start_idx = idx * self.block_size
        end_idx = start_idx + self.block_size + 1
        chunk = self.text[start_idx:end_idx]
        idxs = self.tokenizer.encode(chunk)
        return {
            'x': idxs[:-1],
            'y': idxs[1:]
        }

    def __len__(self):
        return (len(self.text)-1) // self.block_size
    

# Create tokenizer, dataset and config


# Instantiate model


# Put on GPUs



# Create optimizer


# Training loop

