import torch
from tqdm import tqdm

get_rand = lambda : torch.rand((1,)).item()

def generate_sequence(T=100, r=0.35):
    xs = ''
    for t in range(T):
        if get_rand() <= r:
            i = int(26 * get_rand())
            x = chr(ord('a') + i)
        else:
            i = t%26
            x = chr(ord('a') + i)
        xs += x
    return xs

if __name__ == '__main__':
    sequences = [generate_sequence() for _ in tqdm(range(100_000))]
    full_text = '\n'.join(sequences)
    with open('noisy_alphabet.txt', 'w') as f:
        f.write(full_text)
