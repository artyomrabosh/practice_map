import torcheval.metrics
import torch
import numpy as np
from tqdm import tqdm
import wandb


def get_lr(optimizer: torch.optim.Optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def val(model: torch.nn.Module, loader: torch.utils.data.Dataloader, Metrics, device):
    losses_val = []
    cpu_device = torch.device('cpu')

    metrics = {Metric().to(cpu_device): 0 for Metric in Metrics}

    with torch.no_grad():
        for input_ids, attention_masks, labels in tqdm(loader):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device, dtype=torch.int64)

            with torch.inference_mode():
                outputs = model(input_ids,
                                attention_mask=attention_masks,
                                labels=labels)
                losses_val.append(outputs.loss.item())
                predictions = outputs.logits.argmax(axis=1)

                for metric in metrics:
                    metric.update(predictions.to(cpu_device), labels.to(cpu_device))

    return np.mean(losses_val), {metric.__class__.__name__: metric.compute() for metric in metrics}


def learning_loop(
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: torch.utils.data.Dataloader,
        val_loader: torch.utils.data.Dataloader,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        min_lr: float = None,
        epochs: int = 10,
        val_every: int = 100,
        Metrics=None,
        device=None,
):
    lrs = []

    for epoch in range(epochs):
        print(f'#{epoch}/{epochs}:')

        model.train()
        losses_tr = []
        for i, (input_ids, attention_masks, labels) in enumerate(tqdm(train_loader)):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device, dtype=torch.int64)
            optimizer.zero_grad()
            outputs = model(input_ids,
                            attention_mask=attention_masks,
                            labels=labels)
            loss = outputs[0]
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses_tr.append(loss.item())

            if i % val_every == 0 and Metrics:
                loss_val, metrics = val(model, val_loader, Metrics, device)
                metrics['loss_val'] = loss_val
                metrics['loss_train'] = np.mean(losses_tr)
                metrics['learning_rate'] = get_lr(optimizer)

                wandb.log(metrics, step=epoch * len(train_loader) + (i + 1) * len(labels))

        if min_lr and get_lr(optimizer) <= min_lr:
            print(f'Learning process ended with early stop after epoch {epoch}')
            break

    return model, optimizer
