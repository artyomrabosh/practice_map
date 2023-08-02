import wandb
import datetime
from hparams import config
from prepare_data import prepare_data
from utils.datasets import classificationDataset
from utils.ml import learning_loop

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig, get_linear_schedule_with_warmup

# logger
#

def start_wandb():
    name = 'test-({})'.format(
        datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    wandb_config = config.copy()

    wandb_run = wandb.init(project='map', name=name, config=wandb_config)
    wandb_run.log_code('.')
    return wandb_run


def main():
    start_wandb()
    prepare_data()
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer'], do_lower_case=False)

    train = classificationDataset("data/sents/train.csv", tokenizer, MAX_LEN=512)
    val = classificationDataset("data/sents/val.csv", tokenizer, MAX_LEN=512)

    dataloader_train = DataLoader(
        dataset=train,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=True,
    )

    dataloader_val = DataLoader(
        dataset=val,
        batch_size=config['batch_size'],
        shuffle=True,
        drop_last=False,
    )

    device = torch.device("cpu")

    model = BertForSequenceClassification.from_pretrained(
        config['model'],
        output_attentions=False,
        output_hidden_states=False
    ).to(device)

    for param in model.bert.parameters():
        param.requires_grad = config['is_finetuning']

    optimizer = AdamW(model.parameters(),
                      lr=config['learning_rate'],
                      eps=1e-8
                      )

    epochs = config['epochs']

    total_steps = len(dataloader_train) * epochs

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config['num_warmup_steps'],
        num_training_steps=total_steps
    )
    wandb.watch(model, log_freq=100)

    model, optimizer, losses = learning_loop(
        model=model,
        optimizer=optimizer,
        train_loader=dataloader_train,
        val_loader=dataloader_val,
        scheduler=scheduler,
        epochs=epochs,
        min_lr=1e-7,
        val_every=100,
        Metrics=[],
        device=device
    )

    torch.save(model.state_dict(), "model.pt")

    with open("run_id.txt", "w+") as f:
        print(wandb.run.id, file=f)


if __name__ == '__main__':
    main()
