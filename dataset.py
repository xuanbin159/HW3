# import some packages you need here

import torch
from torch.utils.data import Dataset

class Shakespeare(Dataset):

    def __init__(self, input_file):
        with open(input_file, 'r', encoding='utf-8') as f:
            self.data = f.read()

        self.chars = sorted(list(set(self.data)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}

        self.char_indices = [self.char_to_idx[ch] for ch in self.data]

        self.seq_length = 80
        self.num_sequences = len(self.char_indices) // self.seq_length

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length

        input_s = torch.tensor(self.char_indices[start_idx:end_idx])
        target_s = torch.tensor(self.char_indices[start_idx+1:end_idx+1])

        return input_s, target_s

if __name__ == '__main__':
    # Test the Shakespeare dataset
    dataset = Shakespeare('shakespeare_train.txt')

    print(f"Total characters: {len(dataset.data)}")
    print(f"Unique characters: {len(dataset.chars)}")
    print(f"sequences: {len(dataset)}")

    # Get a sample sequence
    input_seq, target_seq = dataset[0]
    print(f"input sequence: {input_seq}")
    print(f"target sequence: {target_seq}")