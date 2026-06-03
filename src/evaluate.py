import torch
import Levenshtein
from src.encoding_decoding import decode_predictions


def decode_ctc_batch(
    outputs: torch.Tensor,
    output_lengths: torch.Tensor,
    idx_to_char: dict,
) -> list[str]:
    batch_size = output_lengths.numel()
    if outputs.size(0) == batch_size:
        outputs = outputs.transpose(0, 1)

    pred_ids = outputs.argmax(dim=2).detach().cpu()
    output_lengths = output_lengths.detach().cpu()

    pred_texts = []
    for i, length in enumerate(output_lengths):
        ids = pred_ids[: length.item(), i].tolist()
        pred_texts.append(decode_predictions(ids, idx_to_char))

    return pred_texts


def decode_target_batch(
    targets: torch.Tensor,
    target_lengths: torch.Tensor,
    idx_to_char: dict,
) -> list[str]:
    targets = targets.detach().cpu()
    target_lengths = target_lengths.detach().cpu()

    true_texts = []

    if targets.dim() == 1:
        offset = 0
        for length in target_lengths:
            length = length.item()
            ids = targets[offset : offset + length].tolist()
            true_texts.append("".join(idx_to_char[int(idx)] for idx in ids if int(idx) != 0))
            offset += length
        return true_texts

    for target, length in zip(targets, target_lengths):
        ids = target[:length.item()].tolist()
        true_texts.append("".join(idx_to_char[int(idx)] for idx in ids if int(idx) != 0))

    return true_texts


def batch_levenshtein_distance(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    output_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
    idx_to_char: dict,
) -> tuple[float, list[int], list[str], list[str]]:
    pred_texts = decode_ctc_batch(outputs, output_lengths, idx_to_char)
    true_texts = decode_target_batch(targets, target_lengths, idx_to_char)

    distances = []    
    for pred_text, true_text in zip(pred_texts, true_texts):
        distances.append(Levenshtein.distance(pred_text, true_text))

    mean_distance = sum(distances) / max(len(distances), 1)
    return mean_distance, distances, pred_texts, true_texts
