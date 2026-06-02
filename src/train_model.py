import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


def _prepare_ctc_inputs(
    outputs: torch.Tensor,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    input_lengths: torch.Tensor,
    target_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = inputs.size(0)

    if outputs.size(0) == batch_size:
        outputs = outputs.permute(1, 0, 2)

    log_probs = F.log_softmax(outputs, dim=2)

    output_time = log_probs.size(0)
    input_time = inputs.size(-1)
    input_lengths = input_lengths.cpu()
    target_lengths = target_lengths.cpu()

    scale = output_time / input_time
    output_lengths = torch.floor(input_lengths.float() * scale).long()
    output_lengths = output_lengths.clamp(min=1, max=output_time)

    targets = targets if targets.dim() == 1 else torch.cat(
        [target[:length.item()] for target, length in zip(targets, target_lengths)]
    )

    return log_probs, targets, output_lengths, target_lengths


def train_model(
    model: nn.Module,
    criterion: nn.CTCLoss,
    optimizer,
    train_loader,
    val_loader,
    epochs: int = 50,
    device='cpu',
    scheduler=None,
    grad_clip: float | None = 5.0,
    save_path: str | None = None,
    restore_best: bool = True,
    verbose: bool = True,
):
    model.to(device)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_state_dict = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets, input_lengths, target_lengths in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            outputs = model(inputs)
            log_probs, targets_1d, output_lengths, target_lengths = _prepare_ctc_inputs(
                outputs=outputs,
                inputs=inputs,
                targets=targets,
                input_lengths=input_lengths,
                target_lengths=target_lengths,
            )

            loss = criterion(
                log_probs,
                targets_1d,
                output_lengths,
                target_lengths,
            )
            loss.backward()

            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets, input_lengths, target_lengths in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)
                log_probs, targets_1d, output_lengths, target_lengths = _prepare_ctc_inputs(
                    outputs=outputs,
                    inputs=inputs,
                    targets=targets,
                    input_lengths=input_lengths,
                    target_lengths=target_lengths,
                )

                loss = criterion(
                    log_probs,
                    targets_1d,
                    output_lengths,
                    target_lengths,
                )
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = copy.deepcopy(model.state_dict())

            if save_path is not None:
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'val_loss': avg_val_loss,
                    },
                    f'{save_path}_{epoch + 1}_best.pt' ,
                )
        else:
            if save_path is not None:
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch + 1,
                        'val_loss': avg_val_loss,
                    },
                    f'{save_path}_{epoch + 1}.pt',
                )

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()

        if verbose:
            print(
                f'Epoch {epoch + 1}/{epochs} | '
                f'train_loss: {avg_train_loss:.4f} | '
                f'val_loss: {avg_val_loss:.4f} | '
                f'best_val_loss: {best_val_loss:.4f}'
            )

    if restore_best and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return model, history
