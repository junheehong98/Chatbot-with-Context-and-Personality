"""
Script for running finetuning on glue tasks.

Largely copied from:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""
import argparse
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss  # 임포트 추가
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import transformers
from transformers import (
    AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)
from tqdm import tqdm

import autoprompt.utils as utils
from autoprompt.popsicle import AutoPopsicle 


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def main(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = AutoConfig.from_pretrained(args.model_name, num_labels=5)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    utils.add_task_specific_tokens(tokenizer)  # 특수 토큰 추가
    model = AutoPopsicle.from_pretrained(args.model_name, config=config)  # 수정된 부분: AutoPopsicle 사용**
    model.to(device)

    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    train_dataset, label_map = utils.load_classification_dataset(
        args.train,
        tokenizer,
        args.field_a,
        args.field_b,
        args.label_field,
        limit=args.limit
    )
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    dev_dataset, _ = utils.load_classification_dataset(
        args.dev,
        tokenizer,
        args.field_a,
        args.field_b,
        args.label_field,
        label_map
    )
    dev_loader = DataLoader(dev_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator)
    test_dataset, _ = utils.load_classification_dataset(
        args.test,
        tokenizer,
        args.field_a,
        args.field_b,
        args.label_field,
        label_map
    )
    test_loader = DataLoader(test_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator)

    if args.bias_correction:
        betas = (0.9, 0.999)
    else:
        betas = (0.0, 0.000)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
        betas=betas
    )

    # Use suggested learning rate scheduler
    num_training_steps = len(train_dataset) * args.epochs // args.bsz
    num_warmup_steps = num_training_steps // 10
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps,
                                                num_training_steps)

    if not args.ckpt_dir.exists():
        logger.info(f'Making checkpoint directory: {args.ckpt_dir}')
        args.ckpt_dir.mkdir(parents=True)
    elif not args.force_overwrite:
        raise RuntimeError('Checkpoint directory already exists.')

    try:
        best_accuracy = 0
        for epoch in range(args.epochs):
            logger.info('Training...')
            model.train()
            avg_loss = utils.ExponentialMovingAverage()
            pbar = tqdm(train_loader)
            for model_inputs, labels in pbar:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)            
                
                
                optimizer.zero_grad()
                 # 수정된 부분: 모델 출력 및 손실 계산 방식 복원
                logits, *_ = model(**model_inputs)
                # print(f"Logits shape: {logits.shape}")  # 로그: logits의 실제 크기를 출력합니다.
                logits = logits.view(-1, args.num_labels, 2)  # (batch_size, num_labels, num_classes)
                # 각 레이블에 대해 손실 계산 및 합산
                # loss = sum(F.cross_entropy(logits[:, i, :], labels[:, i]) for i in range(args.num_labels)) / args.num_labels
                # loss_fct = CrossEntropyLoss(label_smoothing=0.1)  # Label Smoothing 적용
                loss_fct = CrossEntropyLoss()
                loss = sum(loss_fct(logits[:, i, :], labels[:, i]) for i in range(args.num_labels)) / args.num_labels

                
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                avg_loss.update(loss.item())
                pbar.set_description(f'loss: {avg_loss.get_metric(): 0.4f}, '
                                     f'lr: {optimizer.param_groups[0]["lr"]: .3e}')

            logger.info('Evaluating...')
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for model_inputs, labels in dev_loader:
                    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                    labels = labels.to(device)                   
                   
                    
                    logits, *_ = model(**model_inputs)
                    logits = logits.view(-1, args.num_labels, 2)  # (batch_size, num_labels, num_classes)
                    preds = torch.argmax(logits, dim=-1)  # 각 레이블에 대한 예측값 계산
                    correct += (preds == labels).sum().item()
                    total += labels.numel()
                accuracy = correct / (total + 1e-13)
            logger.info(f'Accuracy: {accuracy : 0.4f}')

            if accuracy > best_accuracy:
                logger.info('Best performance so far.')
                model.save_pretrained(args.ckpt_dir)
                tokenizer.save_pretrained(args.ckpt_dir)
                best_accuracy = accuracy
    except KeyboardInterrupt:
        logger.info('Interrupted...')

    logger.info('Testing...')
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for model_inputs, labels in test_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            
            logits, *_ = model(**model_inputs)
            logits = logits.view(-1, args.num_labels, 2)  # (batch_size, num_labels, num_classes)
            preds = torch.argmax(logits, dim=-1)  # 각 레이블에 대한 예측값 계산
            correct += (preds == labels).sum().item()
            total += labels.numel()
        accuracy = correct / (total + 1e-13)
    logger.info(f'Accuracy: {accuracy : 0.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--train', type=Path)
    parser.add_argument('--dev', type=Path)
    parser.add_argument('--test', type=Path)
    parser.add_argument('--field-a', type=str)
    parser.add_argument('--field-b', type=str, default=None)
    parser.add_argument('--label-field', nargs='+', type=str, default=['label1', 'label2', 'label3', 'label4', 'label5'])

    parser.add_argument('--ckpt-dir', type=Path, default=Path('ckpt/'))
    parser.add_argument('--num-labels', type=int, default=5)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--bias-correction', action='store_true')
    parser.add_argument('-f', '--force-overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
