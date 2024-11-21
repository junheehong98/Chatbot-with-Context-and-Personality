import time
import argparse
import json
import logging
from pathlib import Path
import random
import itertools


import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer
from tqdm import tqdm
from autoprompt.popsicle import AutoPopsicle

import autoprompt.utils as utils


logger = logging.getLogger(__name__)


class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


class PredictWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """
    def __init__(self, model, num_labels):
        self.num_labels = num_labels  # num_labels 속성 추가
        self.num_classes = 2  # 클래스 수 = 2
        self._model = model      
       

    def __call__(self, model_inputs, trigger_ids):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')        
        predict_mask = model_inputs.pop('predict_mask')

        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        
        logits, *_ = self._model(**model_inputs)

        # 빈 logits 검증
        if logits.size(0) == 0:
            logger.error("Empty logits tensor encountered. Verify input data and preprocessing.")
            raise ValueError("Logits tensor is empty.")


        logger.debug(f"Logits raw shape: {logits.size()}")


        # logger.debug(f" 1111 Input Raw logits shape: {logits.size()}")
        # logger.debug(f"  1111 Num labels: {self.num_labels}")  # num_labels 확인
        # logger.debug(f"  1111 Last dimension (logits.size(-1)): {logits.size(-1)}")  # 마지막 차원 확인
        # logger.debug(f" 1111 Predict mask shape before processing: {predict_mask.size()}")


        # logger.debug(f"Logits raw shape: {logits.size()}")
        # logger.debug(f"Trigger mask shape: {trigger_mask.size() if trigger_mask is not None else None}")
        # logger.debug(f"Predict mask shape: {predict_mask.size() if predict_mask is not None else None}")



        # predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), self.num_labels, -1)
        # predict_logits = logits  # 모델의 출력을 그대로 사용
        
        # logits을 (batch_size, num_labels, num_classes)로 변환
        # predict_logits = logits.view(logits.size(0), self.num_labels, 2)

        # # 출력 크기 동적 변환
        # if logits.size(-1) % self.num_labels == 0:
        #     num_classes = logits.size(-1) // self.num_labels
        #     logits = logits.view(logits.size(0), self.num_labels, num_classes)
        # else:
        #     raise ValueError(f"Unexpected logits shape: {logits.size()}")
        # Check if logits match expected shape
        
        '''
        # 예상 로짓 크기 확인
        if logits.size(-1) % self.num_labels != 0:
            raise ValueError(
                f"Unexpected logits shape: {logits.size()}, "
                f"expected last dimension to be divisible by num_labels={self.num_labels}."
            )

        num_classes = logits.size(-1) // self.num_labels
        logits = logits.view(logits.size(0), self.num_labels, num_classes)
        '''
        # if predict_mask is not None:
        #     if predict_mask.dim() < logits.dim():
        #         predict_mask = predict_mask.unsqueeze(-1).expand_as(logits)
            
        #     try:
        #         # Predict mask로 logits 선택
        #         predict_logits = logits.masked_select(predict_mask).view(logits.size(0), -1)
        #     except RuntimeError as e:
        #         logger.error(f"Masking failed. Predict mask shape: {predict_mask.size()}, Logits shape: {logits.size()}")
        #         raise e
    
            # predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        # else:
        #     predict_logits = logits

        # Reshape logits to [batch_size, num_labels, num_classes]


        # 로짓 크기 확인 및 변환
        batch_size = logits.size(0)

        # 크기 검증
        if logits.size(-2) != self.num_labels or logits.size(-1) != self.num_classes:
            raise ValueError(
                f"Logits shape mismatch: {logits.size()}, expected (batch_size, {self.num_labels}, {self.num_classes})."
            )



        # logger.debug(f"Logits shape before: {logits.size()}")
        # logits = logits.view(-1, self.num_labels, self.num_classes)
        # logger.debug(f"Logits shape after: {logits.size()}")
        # logger.debug(f"Logits will be reshaped to aft: (batch_size={batch_size}, num_labels={self.num_labels}, num_classes={num_classes})")
        # predict_mask 크기 조정
        if predict_mask.size(1) != logits.size(1):
            predict_mask = predict_mask[:, :logits.size(1)]

        if predict_mask.dim() < logits.dim():
            predict_mask = predict_mask.unsqueeze(-1).expand_as(logits)

        if predict_mask.size() != logits.size():
            raise ValueError(
                f"Mismatch between predict_mask size {predict_mask.size()} and logits size {logits.size()}"
            )
        
        # Predict mask 적용
        if predict_mask is not None:
            if predict_mask.dim() < logits.dim():
                predict_mask = predict_mask.unsqueeze(-1).expand_as(logits)
            logits = logits.masked_select(predict_mask).view(-1, self.num_labels, self.num_classes)

        
        return logits



        # # Reshape logits if necessary
        # if len(logits.size()) == 3:  # (batch_size, num_labels, num_classes)
        #     return logits
        # elif len(logits.size()) == 2:  # Handle unexpected (batch_size, num_classes)
        #     logits = logits.view(logits.size(0), 5, 2)  # (batch_size, num_labels, num_classes)
        #     logger.debug(f"Logits reshaped to: {logits.size()}")
        #     return logits
        # else:
        #     raise ValueError(f"Unexpected logits shape: {logits.size()}")


        # if len(logits.size()) != 3:
        #     raise ValueError(f"Expected logits to have 3 dimensions [batch_size, num_labels, num_classes], but got {logits.size()}")

        


        # predict_logits = logits.view(logits.size(0), 5, 2)  # [batch_size, num_labels, num_classes]
        # logger.debug(f"Logits shape after reshape: {predict_logits.size()}")
        # print(f"Logits shape after reshape: {predict_logits.size()}")
        # # predict_logits = logits.view(logits.size(0), -1)
        # return predict_logits


class AccuracyFn:
    """
    Computing the accuracy when a label is mapped to multiple tokens is difficult in the current
    framework, since the data generator only gives us the token ids. To get around this we
    compare the target logp to the logp of all labels. If target logp is greater than all (but)
    one of the label logps we know we are accurate.
    """
    def __init__(self, tokenizer, label_map, device, num_labels, tokenize_labels=False):
        self._all_label_ids = []
        self._pred_to_label = []
        self.num_labels = num_labels
        logger.info(label_map)
        for label, label_tokens in label_map.items():
            self._all_label_ids.append(utils.encode_label(tokenizer, label_tokens, tokenize_labels).to(device))
            self._pred_to_label.append(label)
        logger.info(self._all_label_ids)

    def __call__(self, predict_logits, gold_label_ids):
        # Get total log-probability for the true label
        gold_logp = get_loss(predict_logits, gold_label_ids , self.num_labels)

        # Get total log-probability for all labels
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]

        # Add up the number of entries where loss is greater than or equal to gold_logp.
        ge_count = all_label_logp.le(gold_logp.unsqueeze(-1)).sum(-1)
        correct = ge_count.le(1)  # less than in case of num. prec. issues

        return correct.float()

    # TODO: @rloganiv - This is hacky. Replace with something sensible.
    def predict(self, predict_logits):
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1), self.num_labels)
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]
        return predictions


def load_pretrained(model_name):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    config = AutoConfig.from_pretrained(model_name)
    model = AutoPopsicle.from_pretrained(model_name, config=config)  # AutoPopsicle 사용
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    utils.add_task_specific_tokens(tokenizer)
    return config, model, tokenizer


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_embeddings(model, config):
    """Returns the wordpiece embedding module."""
    base_model = getattr(model, config.model_type)
    embeddings = base_model.embeddings.word_embeddings
    return embeddings


def evaluation_fn(predict_logits, gold_labels):
    # preds = (torch.sigmoid(predict_logits) >= 0.5).float()

    preds = torch.argmax(predict_logits, dim=1)
    correct = (preds == gold_labels).float()
    accuracy = correct.mean()
    return accuracy


def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        # gradient_dot_embedding_matrix = torch.matmul(
        #     # embedding_matrix,
        #     # averaged_grad
            
        #     averaged_grad.view(-1, embedding_matrix.size(1)), embedding_matrix.t()
        # )

        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad.view(-1, 1)
        ).squeeze(1)


        if filter is not None:
            gradient_dot_embedding_matrix -= filter
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids


def replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask):
    """Replaces the trigger tokens in input_ids."""
    out = model_inputs.copy()
    input_ids = model_inputs['input_ids']
    trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1)
    try:
        filled = input_ids.masked_scatter(trigger_mask, trigger_ids)
    except RuntimeError:
        filled = input_ids
    out['input_ids'] = filled
    return out


def save_best_trigger_tokens(trigger_tokens, scores, combination, output_dir='trigger_tokens'):
    """각 조합의 최적 트리거 토큰을 JSON 파일로 저장합니다."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)  # 디렉토리 생성
    file_name = '_'.join(map(str, combination))
    file_path = Path(output_dir) / f'best_trigger_tokens_{file_name}.json'
    data = {
        'best_trigger_tokens': trigger_tokens,
        'best_score': scores
    }
    with open(file_path, 'w') as f:
        json.dump(data, f)

    logger.info(f'Trigger tokens saved for combination {combination} to {file_path}')
    
    
    
def create_label_combinations():
    """2^5 가지의 레이블 조합을 생성합니다 (각 레이블은 0, 1, 2의 값을 가집니다)."""
    label_values = [0, 1]  # 실제 데이터의 레이블은 0, 1로 저장되어 있음
    return list(itertools.product(label_values, repeat=5))


    
    
def load_best_trigger_tokens(file_path):
    """저장된 JSON 파일에서 최적 트리거 토큰을 불러옵니다."""
    with open(file_path, 'r') as f:
        data = json.load(f)
        trigger_tokens = data['best_trigger_tokens']
        score = data['best_score']
    
    logger.info(f'Trigger tokens loaded from {file_path}')
    return trigger_tokens, score


def apply_trigger_to_system_prompt(model_inputs, trigger_tokens, tokenizer):
    """트리거 토큰을 시스템 프롬프트에 적용합니다."""
    trigger_ids = tokenizer.convert_tokens_to_ids(trigger_tokens)
    trigger_ids = torch.tensor(trigger_ids).unsqueeze(0)
    
    # 기존 model_inputs에서 trigger_mask를 사용하여 트리거 토큰을 교체합니다.
    trigger_mask = model_inputs['trigger_mask']
    model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
    
    return model_inputs




def get_loss(predict_logits, label_ids, num_labels):
    logger.debug(f"Predict Logits Shape: {predict_logits.size()}, Values: {predict_logits[:2]}")
    logger.debug(f"Label IDs Shape: {label_ids.size()}, Values: {label_ids[:2]}")


    # if len(predict_logits.size()) == 2:  # (batch_size, num_classes)
    #     batch_size, num_classes = predict_logits.size()
    #     if num_classes % num_labels != 0:
    #         raise ValueError(f"[ERROR] Logits classes {num_classes} not divisible by num_labels {num_labels}.")
    #     num_classes_per_label = num_classes // num_labels
    #     predict_logits = predict_logits.view(batch_size, num_labels, num_classes_per_label)
    # print(f"[DEBUG] Reshaped logits: {predict_logits.size()}")
    # logger.debug(f"Predict logits shape: {predict_logits.size()}")





    '''
    predict_logp = F.log_softmax(predict_logits, dim=-1)
    target_logp = predict_logp.gather(-1, label_ids)
    target_logp = target_logp - 1e32 * label_ids.eq(0)  # Apply mask
    target_logp = torch.logsumexp(target_logp, dim=-1)
    '''
    # loss = sum(F.cross_entropy(predict_logits[:, i, :], label_ids[:, i]) for i in range(num_labels)) / num_labels
    
    # predict_logits와 label_ids의 차원에 맞게 손실 계산
    # loss = F.cross_entropy(predict_logits.view(-1, predict_logits.size(-1)), label_ids.view(-1))




    # if predict_logits.shape != label_ids.shape:
    #     raise ValueError(f"Mismatch between logits shape {predict_logits.shape} and label shape {label_ids.shape}")
                  

    # # predict_logits = predict_logits.squeeze(-1)
    # loss_fn = torch.nn.BCEWithLogitsLoss()
    # loss = loss_fn(predict_logits, label_ids.float())






    # # logits과 labels의 차원을 맞춤
    # batch_size, num_labels, num_classes = predict_logits.size()
    # logits_flat = predict_logits.view(batch_size * num_labels, num_classes)
    # labels_flat = label_ids.view(batch_size * num_labels)

    # loss_fn = torch.nn.CrossEntropyLoss()
    # loss = loss_fn(logits_flat, labels_flat)


    '''
    if len(predict_logits.size()) == 3:  # (batch_size, num_labels, num_classes)
        batch_size, num_labels, num_classes = predict_logits.size()
        logits_flat = predict_logits.view(batch_size * num_labels, num_classes)
        labels_flat = label_ids.view(batch_size * num_labels)
    elif len(predict_logits.size()) == 2:  # (batch_size, num_classes)
        logits_flat = predict_logits
        labels_flat = label_ids
    else:
        raise ValueError(f"Unexpected logits shape: {predict_logits.size()}")
    # Logits와 Labels 크기 불일치 확인
    if logits_flat.size(0) != labels_flat.size(0):
        raise ValueError(
            f"Mismatch between logits_flat batch size ({logits_flat.size(0)}) and labels_flat ({labels_flat.size(0)})."
        ) '''
    

    '''
    # Check if logits and labels match the expected dimensions
    if predict_logits.size(1) != num_labels or predict_logits.size(2) != 2:
        logger.error(f"Mismatch: Logits shape {predict_logits.size()}, Labels shape {label_ids.size()}")
        raise ValueError(f"Logits shape {predict_logits.size()} does not match expected dimensions (batch_size, {num_labels}, 2).")

    # Reshape for loss calculation
    logits_flat = predict_logits.view(-1, 2)
    labels_flat = label_ids.view(-1)
    '''

    # Reshape for loss calculation
    batch_size, num_labels_pred, num_classes = predict_logits.size()
    if num_labels_pred != num_labels:
        raise ValueError(
            f"Mismatch between logits num_labels ({num_labels_pred}) and expected num_labels ({num_labels})."
        )
    logits_flat = predict_logits.view(batch_size * num_labels, num_classes)
    labels_flat = label_ids.view(batch_size * num_labels)


    
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits_flat, labels_flat)

    return loss
    # return -target_logp


def isupper(idx, tokenizer):
    """
    Determines whether a token (e.g., word piece) begins with a capital letter.
    """
    _isupper = False
    # We only want to check tokens that begin words. Since byte-pair encoding
    # captures a prefix space, we need to check that the decoded token begins
    # with a space, and has a capitalized second character.
    if isinstance(tokenizer, transformers.GPT2Tokenizer):
        decoded = tokenizer.decode([idx])
        if decoded[0] == ' ' and decoded[1].isupper():
            _isupper = True
    # For all other tokenization schemes, we can just check the first character
    # is capitalized.
    elif tokenizer.decode([idx])[0].isupper():
            _isupper = True
    return _isupper

def create_token_filter(tokenizer, args, label_map, train_loader):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    filter = torch.zeros(len(tokenizer), dtype=torch.float32, device=device)
    if args.filter:
        logger.info('Filtering label tokens.')
        if label_map:
            for label_tokens in label_map.values():
                label_ids = utils.encode_label(tokenizer, label_tokens).unsqueeze(0)
                filter[label_ids] = -1e32
        else:
            for _, label_ids in train_loader.dataset:
                filter[label_ids] = -1e32
        logger.info('Filtering special tokens and capitalized words.')
        for word, idx in tokenizer.get_vocab().items():
            if len(word) == 1 or idx >= len(tokenizer):
                continue
            # Filter special tokens.
            if idx in tokenizer.all_special_ids:
                logger.debug('Filtered: %s', word)
                filter[idx] = -1e32
            # Filter capitalized words (lazy way to remove proper nouns).
            if isupper(idx, tokenizer):
                logger.debug('Filtered: %s', word)
                filter[idx] = -1e32
    return filter




def load_datasets(args, templatizer, collator):
    logger.info('Loading training and validation datasets')
    if args.perturbed:
        train_dataset = utils.load_augmented_trigger_dataset(args.train, templatizer, limit=args.limit)
    else:
        train_dataset = utils.load_trigger_dataset(args.train, templatizer, use_ctx=args.use_ctx, limit=args.limit)




    # # 데이터셋 검증 (샘플 구조 확인)
    # for i, sample in enumerate(train_dataset):
    #     if isinstance(sample, tuple):  # Tuple structure: (input_ids, labels)
    #         input_ids, labels = sample
    #     elif isinstance(sample, dict):  # Dictionary structure
    #         labels = sample['labels']
    #     else:
    #         raise ValueError(f"Unexpected sample structure: {type(sample)}")

    #     if len(labels) != args.num_labels:
    #         logger.warning(f"Sample {i} has invalid label size: {len(labels)}")
    #         continue


    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    if args.perturbed:
        dev_dataset = utils.load_augmented_trigger_dataset(args.dev, templatizer)
    else:
        dev_dataset = utils.load_trigger_dataset(args.dev, templatizer, use_ctx=args.use_ctx)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
    logger.info('Datasets loaded successfully')


    
    return train_loader, dev_loader

def find_and_evaluate_triggers(model, tokenizer, templatizer, predictor, embedding_gradient, device, filter, evaluation_fn, train_loader, dev_loader, args, embeddings):
    logger.info('Initializing trigger tokens')
    # 초기 트리거 토큰 설정
    if args.initial_trigger:
        trigger_ids = tokenizer.convert_tokens_to_ids(args.initial_trigger)
        logger.debug(f'Initial trigger: {args.initial_trigger}')
        logger.debug(f'Trigger ids: {trigger_ids}')
        assert len(trigger_ids) == templatizer.num_trigger_tokens
    else:
        trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
    
    # 트리거 토큰의 길이와 템플릿의 [T] 개수가 맞는지 확인하는 부분
    assert len(trigger_ids) == templatizer.num_trigger_tokens, \
        f"Trigger token length mismatch. Expected {templatizer.num_trigger_tokens}, but got {len(trigger_ids)}"

    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)
    best_trigger_ids = trigger_ids.clone()

    # 초기 평가
    logger.info('Initial evaluation of triggers')
    dev_metric = evaluate_triggers(predictor, dev_loader, evaluation_fn, trigger_ids, device)
    

    best_dev_metric = dev_metric

    logger.info('Starting trigger search...')
    start = time.time()


    for i in range(args.iters):
        logger.info(f'Iteration: {i}')
        averaged_grad = accumulate_gradients(model, predictor, train_loader, trigger_ids, embedding_gradient, args)

        # 후보자 생성 및 평가
        logger.info('Evaluating Candidates')
        token_to_flip = random.randrange(templatizer.num_trigger_tokens)
        best_candidate_score, best_candidate_idx, candidates = evaluate_candidates(
            model, predictor, train_loader, averaged_grad, trigger_ids, args, tokenizer, embeddings, filter, token_to_flip, evaluation_fn
        )

        if best_candidate_score > best_dev_metric:
            logger.info('Better trigger detected.')
            trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
            best_dev_metric = best_candidate_score
        else:
            logger.info('No improvement detected. Skipping evaluation.')
            continue

        # 평가 단계
        logger.info('Evaluating dev dataset with new triggers')
        dev_metric = evaluate_triggers(predictor, dev_loader, evaluation_fn, trigger_ids, device)

        if dev_metric > best_dev_metric:
            logger.info('Best performance so far')
            best_trigger_ids = trigger_ids.clone()
            best_dev_metric = dev_metric

    logger.info(f'Finished trigger search. Best dev metric: {best_dev_metric}')
    return best_trigger_ids, best_dev_metric


def accumulate_gradients(model, predictor, train_loader, trigger_ids, embedding_gradient, args):
    """
    Accumulates gradients over multiple batches and computes averaged gradients.
    """
    model.train()
    embedding_gradient._stored_gradient = None  # 초기화

    train_iter = iter(train_loader)
    for step in range(args.accumulation_steps):
        try:
            model_inputs, labels = next(train_iter)
        except StopIteration:
            logger.warning('Not enough data for the specified number of accumulation steps.')
            break

        model_inputs = {k: v.to(trigger_ids.device) for k, v in model_inputs.items()}
        labels = labels.to(trigger_ids.device)
        
        
        # Trigger 토큰을 교체하고 예측 수행
        predict_logits = predictor(model_inputs, trigger_ids)
        # `predict_logits`이 (batch_size, num_labels) 형태인지 확인

        #

        logger.debug(f"Step {step}: Predict logits shape: {predict_logits.size()}, Labels shape: {labels.size()}")

        #



        # predict_logits = predict_logits.view(-1, args.num_labels, 3)  # 로짓의 형태 조정

        # 손실 계산 및 역전파
        # 각 레이블별로 손실을 계산하고 평균
        # loss = sum(F.cross_entropy(predict_logits[:, i, :], labels[:, i]) for i in range(args.num_labels)) / args.num_labels
        # 손실 계산 및 역전파
        # loss = 0
        # for i in range(args.num_labels):
        #     # print(f"predict_logits[:, {i}, :].shape: {predict_logits[:, i, :].shape}")
        #     # print(f"labels[:, {i}].shape: {labels[:, i].shape}")
        #     loss += F.cross_entropy(predict_logits[:, i, :], labels[:, i])
        # loss = loss / args.num_labels

        # 손실 계산
        # loss = get_loss(predict_logits, labels)
        loss = get_loss(predict_logits, labels, args.num_labels)   
        logger.debug(f"Step {step}: Loss: {loss.item()}")
        
        
        model.zero_grad()  # 그라디언트 초기화
        loss.backward()

        # 그라디언트 저장
        grad = embedding_gradient.get()
        if grad is None:
            raise ValueError("Gradient not found. Ensure the backward hook is properly registered.")
        logger.debug(f"Step {step}: Grad shape before masking: {grad.size()}")
        logger.debug(f"Step {step}: Trigger mask shape: {model_inputs['trigger_mask'].size()}")
       
        # Trigger mask 크기 확인 및 확장
        trigger_mask = model_inputs['trigger_mask'].unsqueeze(-1)
        if trigger_mask.size() != grad.size():
            logger.error(f"Mismatch between trigger_mask size {trigger_mask.size()} and grad size {grad.size()}.")
            trigger_mask = trigger_mask.expand_as(grad)  # 다시 크기를 확장
            if trigger_mask.size() != grad.size():
                raise ValueError(f"Expanded trigger_mask size {trigger_mask.size()} still does not match grad size {grad.size()}.")
        logger.debug(f"Step {step}: Trigger mask shape: {trigger_mask.size()}")


        # Trigger mask 크기와 그라디언트 크기 검증
        if trigger_mask.size() != grad.size():
            logger.error(
                f"[Error] Trigger mask size {trigger_mask.size()} does not match grad size {grad.size()}. "
                "This might cause gradient alignment issues. Please verify the input template and masking logic."
            )
            raise ValueError(
                f"Mismatch between trigger mask and gradient dimensions: "
                f"Trigger mask: {trigger_mask.size()}, Gradient: {grad.size()}"
            )
                    
        


        # grad = grad.masked_select(selection_mask).view(bsz, -1, emb_dim)
        grad = grad.masked_select(trigger_mask).view(-1, grad.size(-1))
        # grad = grad.sum(dim=0)  # Sum over batch dimension
        logger.debug(f"Step {step}: Grad shape after masking: {grad.size()}")
        

        # # 그라디언트 저장
        # current_grad = embedding_gradient.get()

        # # 그라디언트 저장
        # if embedding_gradient._stored_gradient is None:
        #     embedding_gradient._stored_gradient = current_grad.clone()
        # else:
        #     embedding_gradient._stored_gradient += current_grad

        

        if embedding_gradient._stored_gradient is None:
            embedding_gradient._stored_gradient = grad.clone()            
        else:   
            if embedding_gradient._stored_gradient.size() != grad.size():
                raise ValueError(
                    f"Stored gradient size mismatch: {embedding_gradient._stored_gradient.size()} vs {grad.size()}."
                )         
            embedding_gradient._stored_gradient += grad
        


    # 평균 그라디언트를 반환

    if embedding_gradient._stored_gradient is None:
        raise ValueError("No gradient stored after accumulation. Check your training loop.")
    

    averaged_grad = embedding_gradient._stored_gradient / args.accumulation_steps
    logger.debug(f"Averaged gradient shape: {averaged_grad.size()}")
    return averaged_grad

def preprocess_labels(row):
    '''
    # Check if all labels are valid integers
    for label in ['Label1', 'Label2', 'Label3', 'Label4', 'Label5']:
        if not row[label].isdigit():
            return None  # Mark the row for exclusion
    # Convert labels to integers
    row = {label: int(row[label]) for label in ['Label1', 'Label2', 'Label3', 'Label4', 'Label5']}
    return row
    '''
    try:
        for label in ['Label1', 'Label2', 'Label3', 'Label4', 'Label5']:
            row[label] = int(row[label])
        return row
    except ValueError:
        return None  # Skip rows with invalid labels





def setup_evaluation_function(tokenizer, args, label_map, device):
    if label_map:
        logger.info('Setting up Accuracy evaluation function')
        evaluation_fn = AccuracyFn(tokenizer, label_map, device, args.num_labels)
    else:
        logger.info('Setting up Loss-based evaluation function')
        evaluation_fn = lambda x, y: -get_loss(x, y, args.num_labels)
    
    # 추가: evaluation_fn이 잘 설정되었는지 로그 출력
    logger.debug(f'Evaluation function set: {evaluation_fn}')
    return evaluation_fn

def print_lama_template(best_trigger_ids, tokenizer, templatizer, args):
    if args.use_ctx:
        model_inputs, label_ids = templatizer({
            'sub_label': '[X]',
            'obj_label': tokenizer.lama_y,
            'context': ''
        })
    else:
        model_inputs, label_ids = templatizer({
            'sub_label': '[X]',
            'obj_label': tokenizer.lama_y,
        })

    lama_template = model_inputs['input_ids']
    lama_template.masked_scatter_(mask=model_inputs['trigger_mask'], source=best_trigger_ids.cpu())
    lama_template.masked_scatter_(mask=model_inputs['predict_mask'], source=label_ids)
    relation = args.train.parent.stem

    if args.use_ctx:
        template = tokenizer.decode(lama_template.squeeze(0)[1:-1]).replace('[SEP] ', '').replace('</s> ', '').replace('[ X ]', '[X]')
    else:
        template = tokenizer.decode(lama_template.squeeze(0)[1:-1]).replace('[ X ]', '[X]')

    out = {'relation': relation, 'template': template}
    print(json.dumps(out))

def evaluate_triggers(predictor, dev_loader, evaluation_fn, trigger_ids, device):
    # total_correct = 0
    # total_count = 0
    total_accuracy = 0
    total_batches = 0


    for model_inputs, labels in tqdm(dev_loader):
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        with torch.no_grad():
            # predict_logits = predictor(model_inputs, trigger_ids)
            # # predict_logits = predict_logits.view(-1, args.num_labels, 3)
            # preds = torch.argmax(predict_logits, dim=-1)
            # correct = (preds == labels).sum().item()
            # total_correct += correct
            # total_count += labels.numel()
            predict_logits = predictor(model_inputs, trigger_ids)

            logger.debug(f"Predict logits shape: {predict_logits.size()}, Labels shape: {labels.size()}")
            # 디버깅: predict_logits 구조 출력
            logger.debug(f"Predict logits shape: {predict_logits.size()}, Values: {predict_logits[:2]}")
            logger.debug(f"Labels shape: {labels.size()}, Values: {labels[:2]}")

           


            # logits에서 예측된 클래스 인덱스 추출
            predictions = torch.argmax(predict_logits, dim=-1)  # (batch_size, num_labels)


            # predict_logits이 (batch_size, num_labels)인지 확인 필요

            # if predict_logits.shape != labels.shape:
            #     raise ValueError(
            #         f"Shape mismatch: logits {predict_logits.shape}, labels {labels.shape}"
            #     )


            # accuracy = evaluation_fn(predict_logits, labels)

            # accuracy = evaluation_fn(logits_flat, labels_flat)
            # Directly calculate accuracy
            
            # 레이블과 예측 비교
            correct = (predictions == labels).float()
            accuracy = correct.mean().item()  # 배치 정확도 계산

        total_accuracy += accuracy
        total_batches += 1

    # dev_metric = total_correct / (total_count + 1e-13)
    dev_metric = total_accuracy / total_batches
    logger.info(f'Dev metric: {dev_metric}')
    return dev_metric

def evaluate_candidates(model, predictor, train_loader, averaged_grad, trigger_ids, args, tokenizer, embeddings, filter, token_to_flip, evaluation_fn):
    """
    Evaluate the candidate tokens to find the best replacement for trigger optimization.
    """
    logger.debug(f"Evaluating candidates for token position {token_to_flip}")
    # 추가: evaluation_fn이 정상적으로 전달되었는지 로그 확인
    logger.debug(f'evaluation_fn is: {evaluation_fn}')
    if evaluation_fn is None:
        raise ValueError("evaluation_fn is not defined.")
    
    
    candidates = hotflip_attack(averaged_grad[token_to_flip],
                                embeddings.weight,
                                increase_loss=False,
                                num_candidates=args.num_cand,
                                filter=filter)
    
    # Ensure candidates and candidate_scores have the same size
    num_candidates = len(candidates)
    # candidate_scores = torch.zeros(args.num_cand, device=trigger_ids.device)
    candidate_scores = torch.zeros(num_candidates, device=trigger_ids.device)

    current_score = 0    
    denom = 0

    train_iter = iter(train_loader)
    pbar = tqdm(range(args.accumulation_steps))
    
    for step in pbar:
        try:
            model_inputs, labels = next(train_iter)
        except StopIteration:
            logger.warning('Insufficient data for number of accumulation steps. Effective batch size will be smaller than specified.')
            break

        model_inputs = {k: v.to(trigger_ids.device) for k, v in model_inputs.items()}
        labels = labels.to(trigger_ids.device)
        
        with torch.no_grad():
            predict_logits = predictor(model_inputs, trigger_ids)

            if predict_logits.size(0) != labels.size(0):
                raise ValueError(
                    f"Mismatch between logits batch size ({predict_logits.size(0)}) and labels ({labels.size(0)})."
                )
            eval_metric = evaluation_fn(predict_logits, labels)

        logger.debug(f"Original trigger eval metric: {eval_metric.item()}")

        # current_score += eval_metric.sum()
        # denom += labels.size(0)

        # for i, candidate in enumerate(candidates):
        #     temp_trigger = trigger_ids.clone()
        #     temp_trigger[:, token_to_flip] = candidate[0]
        #     with torch.no_grad():
        #         predict_logits = predictor(model_inputs, temp_trigger)
        #         eval_metric = evaluation_fn(predict_logits, labels)
        #     candidate_scores[i] += eval_metric.sum()

        for i, candidate in enumerate(candidates):
            # Replace token in trigger_ids with the candidate
            temp_trigger = trigger_ids.clone()
            temp_trigger[:, token_to_flip] = candidate
            with torch.no_grad():
                predict_logits = predictor(model_inputs, temp_trigger)
                candidate_eval_metric = evaluation_fn(predict_logits, labels)

            # Accumulate scores for each candidate
            candidate_scores[i] += candidate_eval_metric.item()



    best_candidate_score = candidate_scores.max()
    best_candidate_idx = candidate_scores.argmax()

    return best_candidate_score, best_candidate_idx, candidates

def run_model(args):

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Loading model, tokenizer, etc.')
    
    '''

    # AutoPopsicle을 사용하여 수정된 BERT 모델 로드
    config = AutoConfig.from_pretrained(args.model_name, num_labels=15)  # 특성 수와 레이블에 맞게 설정
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)    
    # 추가: 트리거 토큰과 예측 토큰을 tokenizer에 추가
    utils.add_task_specific_tokens(tokenizer)        
    model = AutoPopsicle.from_pretrained(args.model_name, config=config)  # AutoPopsicle 로드
    model.to(device)
    '''
    
    
     # 학습된 모델 로드
    config = AutoConfig.from_pretrained(str(args.ckpt_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(args.ckpt_dir))
    utils.add_task_specific_tokens(tokenizer)
    model = AutoPopsicle.from_pretrained(str(args.ckpt_dir), config=config)  # AutoPopsicle 로드
    # 모델의 임베딩 크기를 토크나이저에 맞게 조정
    model.resize_token_embeddings(len(tokenizer))
    model.to(device) 


    
    embeddings = get_embeddings(model, config)
    embedding_gradient = GradientStorage(embeddings)
    predictor = PredictWrapper(model, args.num_labels)

    if args.label_map is not None:
        label_map = json.loads(args.label_map)
        logger.info(f"Label map: {label_map}")
    else:
        label_map = None
        logger.info('No label map')

    templatizer = utils.TriggerTemplatizer(
        args.template,
        config,
        tokenizer,
        label_map=label_map,
        label_field=['label1', 'label2', 'label3', 'label4', 'label5'],
        tokenize_labels=args.tokenize_labels,
        add_special_tokens=False,
        use_ctx=args.use_ctx
    )

   
    # 데이터 로딩
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    train_loader, dev_loader = load_datasets(args, templatizer, collator)

    # 필터링 단계
    logger.info('Filtering unwanted tokens...')
    filter = create_token_filter(tokenizer, args, label_map, train_loader)

    # 평가 함수 설정
    evaluation_fn = setup_evaluation_function(tokenizer, args, label_map, device)



    # 모델 출력 디버깅
    for model_inputs, labels in train_loader:
        logger.debug(f"Labels shape: {labels.size()}")
        logger.debug(f"Input IDs shape: {model_inputs['input_ids'].size()}")

        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)

        # Trigger token 준비
        trigger_ids = torch.tensor(
            [tokenizer.mask_token_id] * templatizer.num_trigger_tokens, 
            device=device
        ).unsqueeze(0)




        # PredictWrapper를 사용하여 logits 얻기
        logits = predictor(model_inputs, trigger_ids)
        
        # 디버깅: 출력 크기 확인
        logger.info(f"Logits shape: {logits.shape}")  # logits 크기 확인
        logger.info(f"Labels shape: {labels.shape}")  # labels 크기 확인

        # Logits을 (batch_size, num_labels, num_classes)로 변환 확인
        try:
            logits = logits.view(-1, args.num_labels, 2)  # num_classes = 2
            logger.info(f"Reshaped logits: {logits.shape}")
        except RuntimeError as e:
            logger.error(f"Error reshaping logits: {e}")
        
        break  # 샘플 1개만 확인





    
    # 템플릿에서 [T]의 개수와 initial_trigger의 개수가 일치하는지 확인
    if args.initial_trigger:
        # args.initial_trigger가 리스트일 경우 처리
        if isinstance(args.initial_trigger, list):
            initial_trigger_length = len(args.initial_trigger)
        else:
            initial_trigger_length = len(args.initial_trigger.split())
    
        expected_trigger_tokens = templatizer.num_trigger_tokens
        assert initial_trigger_length == expected_trigger_tokens, \
            f"Initial trigger length ({initial_trigger_length}) does not match number of [T] tokens ({expected_trigger_tokens})."
        

        trigger_ids = tokenizer.convert_tokens_to_ids(args.initial_trigger)


    
     # 모든 레이블 조합에 대해 트리거를 탐색
    label_combinations = create_label_combinations()
    start_index = args.start_index if args.start_index < len(label_combinations) else 0
    combinations_to_process = label_combinations[start_index:]


    logger.info(f'Starting from index {start_index} out of {len(label_combinations)} combinations.')


    
    for combination in tqdm(combinations_to_process, desc="Processing label combinations"):
        logger.info(f'Evaluating combination: {combination}')
        
        # 트리거 탐색 및 평가
        best_trigger_ids, best_dev_metric = find_and_evaluate_triggers(
            model, tokenizer, templatizer, predictor, embedding_gradient, device,
            filter, evaluation_fn, train_loader, dev_loader, args, embeddings
        )
        
        # 최적 트리거 토큰을 텍스트로 변환
        best_trigger_tokens = tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))
        logger.info(f'Best tokens for combination {combination}: {best_trigger_tokens}')
        logger.info(f'Best dev metric for combination {combination}: {best_dev_metric}')

        # 결과를 JSON 파일로 저장
        save_best_trigger_tokens(best_trigger_tokens, best_dev_metric, combination, output_dir='trigger_tokens')

    logger.info('Trigger token search and saving completed for all combinations.')
    
    '''

    # 트리거 탐색 및 평가 함수 호출
    logger.debug(f'Passing evaluation function to find_and_evaluate_triggers: {evaluation_fn}')
    best_trigger_ids, best_dev_metric = find_and_evaluate_triggers(
        model, tokenizer, templatizer, predictor, embedding_gradient, device, filter, evaluation_fn, train_loader, dev_loader, args, embeddings
    )
    

    # 결과 출력
    best_trigger_tokens = tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))
    logger.info(f'Best tokens: {best_trigger_tokens}')
    logger.info(f'Best dev metric: {best_dev_metric}')
    '''

    if args.print_lama:
        print_lama_template(best_trigger_ids, tokenizer, templatizer, args)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True, help='Train data path')
    parser.add_argument('--dev', type=Path, required=True, help='Dev data path')
    parser.add_argument('--template', type=str, help='Template string')
    parser.add_argument('--label-map', type=str, default=None, help='JSON object defining label map')
    parser.add_argument('--ckpt-dir', type=Path, required=True, help='Checkpoint directory containing the fine-tuned model')
    parser.add_argument('--num-labels', type=int, default=5, help='Number of labels')


    # LAMA-specific
    parser.add_argument('--tokenize-labels', action='store_true',
                        help='If specified labels are split into word pieces.'
                             'Needed for LAMA probe experiments.')
    parser.add_argument('--filter', action='store_true',
                        help='If specified, filter out special tokens and gold objects.'
                             'Furthermore, tokens starting with capital '
                             'letters will not appear in triggers. Lazy '
                             'approach for removing proper nouns.')
    parser.add_argument('--print-lama', action='store_true',
                        help='Prints best trigger in LAMA format.')

    parser.add_argument('--initial-trigger', nargs='+', type=str, default=None, help='Manual prompt')
    parser.add_argument('--label-field', nargs='+', type=str, default=['label1', 'label2', 'label3', 'label4', 'label5'],
                        help='Name of the label field')

    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--eval-size', type=int, default=256, help='Eval size')
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--accumulation-steps', type=int, default=10)
    # parser.add_argument('--model-name', type=str, default='bert-base-cased',
    #                     help='Model name passed to HuggingFace AutoX classes.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--use-ctx', action='store_true',
                        help='Use context sentences for relation extraction only')
    parser.add_argument('--perturbed', action='store_true',
                        help='Perturbed sentence evaluation of relation extraction: replace each object in dataset with a random other object')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num-cand', type=int, default=10)
    parser.add_argument('--sentence-size', type=int, default=50)

    parser.add_argument('--debug', action='store_true')

    # ArgumentParser에 추가
    parser.add_argument('--start-index', type=int, default=0, help='Start index for label combination processing')


    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    run_model(args)
