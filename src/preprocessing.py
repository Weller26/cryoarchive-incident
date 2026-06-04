import os
import torch
import torchaudio
import torchaudio.transforms as T
import torchaudio.functional as F
from tqdm.notebook import tqdm
import pandas as pd

def _dynamic_frequency_masking(spec, window=3):
    freq_energies = spec.mean(dim=2)

    peak_freq_idx = freq_energies.argmax(dim=1).item()

    lower_bound = max(0, peak_freq_idx - window)
    upper_bound = min(spec.shape[1], peak_freq_idx + window + 1)
    
    mask = torch.zeros_like(spec)
    mask[:, lower_bound:upper_bound, :] = 1.0

    min_val = spec.min()
    clean_spec = torch.where(mask == 1.0, spec, min_val)
    
    return clean_spec

def _thresholding(spec, threshold_percentile=0.9):
    threshold_val = torch.quantile(spec, threshold_percentile)

    min_val = spec.min()
    clean_spec = torch.where(spec < threshold_val, min_val, spec)
    
    return clean_spec

def transform_wav_to_spectrogram(
    audio_path: str,
    spectrogram,
    amplitude_to_db,
    target_sr=8000,
) -> torch.Tensor:
    waveform, sr = torchaudio.load(audio_path)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    if sr != target_sr:
        waveform = F.resample(waveform, orig_freq=sr, new_freq=target_sr)

    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val

    spec = spectrogram(waveform)
    spec = amplitude_to_db(spec)
    
    spec = spec[:, :70, :]
 
    spec = _dynamic_frequency_masking(spec)
    spec = _thresholding(spec)
    
    spec = (spec - spec.mean()) / (spec.std() + 1e-8)
    
    return spec

def preprocessing_spectrograms(
    csv_path, 
    source_dir, 
    target_dir,
    n_fft=512,
    hop_length=128,
    target_sr=8000, 
    top_db=80
):
    os.makedirs(target_dir, exist_ok=True)
    df = pd.read_csv(csv_path)
    
    spectrogram = T.Spectrogram(n_fft=n_fft, hop_length=hop_length)
    amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=top_db)
    
    for filename in tqdm(df['filename']):
        audio_path = os.path.join(source_dir, filename)

        pt_filename = filename.replace('.wav', '.pt')
        save_path = os.path.join(target_dir, pt_filename)

        if os.path.exists(save_path):
            continue

        spec = transform_wav_to_spectrogram(
            audio_path,
            spectrogram,
            amplitude_to_db,
            target_sr=target_sr
        )
        
        torch.save(spec, save_path)