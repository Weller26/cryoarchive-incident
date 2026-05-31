import torch

def build_char_idx_dict(alphabet: str):
    char_to_idx = {}
    idx_to_char = {}
    
    char_to_idx['<blank>'] = 0
    idx_to_char[0] = '<blank>'
    
    for idx, char in enumerate(alphabet, start=1):
        char_to_idx[char] = idx
        idx_to_char[idx] = char
        
    return char_to_idx, idx_to_char

def encode_text(text: str, char_to_idx: dict):
    text = text.upper()
    indices = [char_to_idx[char] for char in text if char in char_to_idx]
    return torch.tensor(indices, dtype=torch.long)

def decode_predictions(predictions, idx_to_char: dict):
    decoded = []
    prev_char = None
    
    for pred in predictions:
        pred = int(pred)
        
        if pred != prev_char:
            if pred != 0:
                decoded.append(idx_to_char[pred])

        prev_char = pred
        
    return ''.join(decoded)