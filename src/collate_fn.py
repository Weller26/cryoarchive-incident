import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    specs, labels, spec_lengths, label_lengths = zip(*batch)
    
    spec_lengths = torch.tensor(spec_lengths, dtype=torch.long)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    
    specs_transposed = [spec.squeeze(0).transpose(0, 1) for spec in specs]
    
    padded_specs = pad_sequence(specs_transposed, batch_first=True)
    padded_specs = padded_specs.transpose(1, 2).unsqueeze(1)
    
    padded_labels = pad_sequence(labels, batch_first=True)
    
    return padded_specs, padded_labels, spec_lengths, label_lengths

def test_collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor]:
    specs, _, spec_lengths, _ = zip(*batch)

    spec_lengths = torch.tensor(spec_lengths, dtype=torch.long)
    specs_transposed = [spec.squeeze(0).transpose(0, 1) for spec in specs]

    padded_specs = pad_sequence(specs_transposed, batch_first=True)

    padded_specs = padded_specs.transpose(1, 2).unsqueeze(1)

    return padded_specs, spec_lengths