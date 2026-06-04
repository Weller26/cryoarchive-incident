from torch.utils.data import Dataset
import torch
import torchaudio.transforms as T

class MorseDataset(Dataset):
    def __init__(
        self,
        file_paths: list[str],
        labels: list[str],
        char_to_idx: dict,
    ):
        super().__init__()
        
        self.file_paths = file_paths
        self.labels = labels
        self.char_to_idx = char_to_idx
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        path = self.file_paths[index]
        text = self.labels[index]
        
        spec = torch.load(path)
        
        label_indices = [self.char_to_idx[c] for c in text]
        label_tensor = torch.tensor(label_indices, dtype=torch.long)
        
        return spec, label_tensor, spec.shape[2], len(label_indices)