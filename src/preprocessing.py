import os
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm.notebook import tqdm
import pandas as pd

def preprocessing_spectrograms(csv_path, source_dir, target_dir, target_sr=8000, top_db=80):
    os.makedirs(target_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    spectrogram = T.Spectrogram(n_fft=512, hop_length=128, power=2.0)
    amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=top_db)
    
    for filename in tqdm(df['filename']):
        audio_path = os.path.join(source_dir, filename)

        pt_filename = filename.replace('.wav', '.pt')
        save_path = os.path.join(target_dir, pt_filename)

        if os.path.exists(save_path):
            continue

        waveform, sr = torchaudio.load(audio_path)
        
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != target_sr:
            waveform = T.Resample(orig_freq=sr, new_freq=target_sr)(waveform)

        max_val = waveform.abs().max()
        if max_val > 0:
            waveform = waveform / max_val

        spec = spectrogram(waveform)
        spec = amplitude_to_db(spec)
        
        spec = spec = (spec - spec.mean()) / (spec.std() + 1e-8)
        
        torch.save(spec, save_path)