from torch.utils.data import Dataset
import torch
import torchaudio
import torchaudio.transforms as T

class MorseDataset(Dataset):
    def __init__(self, file_paths: list[str], labels: list[str], char_to_idx: dict, target_sr=8000, top_db=80):
        super().__init__()
        
        self.file_paths = file_paths
        self.labels = labels
        self.char_to_idx = char_to_idx
        self.target_sr = target_sr
        self.top_db = top_db
        
        self.spectogram = T.Spectrogram(n_fft=512, hop_length=128)
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=top_db)
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, index):
        path = self.file_paths[index]
        text = self.labels[index]
        
        waveform, sr = torchaudio.load(path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.target_sr:
            waveform = T.Resample(orig_freq=sr, new_freq=self.target_sr)(waveform)

        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        spec = self.spectogram(waveform)
        spec = self.amplitude_to_db(spec)
        
        spec = spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        
        label_indices = [self.char_to_idx[c] for c in text]
        label_tensor = torch.tensor(label_indices, dtype=torch.long)
        
        return spec, label_tensor, spec.shape[2], len(label_indices)