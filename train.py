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
ds = CharDataset('noisy_alphabet.txt', 1024)
cfg = Config(
    vocab_size=ds.tokenizer.vocab_size, 
    block_size=BLOCK_SIZE,
    n_layer=12
)

# Instantiate model
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(f'Using {device} as default rank_0')
model = GPT(cfg)
print(f'Initialised model with {cfg.n_layer} layers')

# Put on GPUs
print('Putting on GPU...')
model = model.to(device)
print('done!')


# Create optimizer
optim = AdamW(
    model.parameters(), 
    lr=LR, 
    betas=(0.9, 0.95)
)
print('Created optimizer')

# Training loop
for epoch in range(EPOCHS):
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
    )
    print('Created dataloader')
    for idx, batch in enumerate(dl):
        batch = {k:v.to(device) for k,v in batch.items()}
        logits, loss = model(batch['x'], targets=batch['y'])
        print(f'Loss for batch {idx}: {loss.item()}')
        loss.backward()
        optim.step()
        optim.zero_grad()
