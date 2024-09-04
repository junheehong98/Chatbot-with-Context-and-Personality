import time
import argparse
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer
from tqdm import tqdm

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
    def __init__(self, model):
        self._model = model

    def __call__(self, model_inputs, trigger_ids):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        predict_mask = model_inputs.pop('predict_mask')
        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        logits, *_ = self._model(**model_inputs)
        predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        return predict_logits


class AccuracyFn:
    """
    Computing the accuracy when a label is mapped to multiple tokens is difficult in the current
    framework, since the data generator only gives us the token ids. To get around this we
    compare the target logp to the logp of all labels. If target logp is greater than all (but)
    one of the label logps we know we are accurate.
    """
    def __init__(self, tokenizer, label_map, device, tokenize_labels=False):
        self._all_label_ids = []
        self._pred_to_label = []
        logger.info(label_map)
        for label, label_tokens in label_map.items():
            self._all_label_ids.append(utils.encode_label(tokenizer, label_tokens, tokenize_labels).to(device))
            self._pred_to_label.append(label)
        logger.info(self._all_label_ids)

    def __call__(self, predict_logits, gold_label_ids):
        # Get total log-probability for the true label
        gold_logp = get_loss(predict_logits, gold_label_ids)

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
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
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
    model = AutoModelWithLMHead.from_pretrained(model_name)
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


def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1,
                   filter=None):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
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


def get_loss(predict_logits, label_ids):
    predict_logp = F.log_softmax(predict_logits, dim=-1)
    target_logp = predict_logp.gather(-1, label_ids)
    target_logp = target_logp - 1e32 * label_ids.eq(0)  # Apply mask
    target_logp = torch.logsumexp(target_logp, dim=-1)
    return -target_logp


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
    filter = torch.zeros(tokenizer.vocab_size, dtype=torch.float32, device='cuda' if torch.cuda.is_available() else 'cpu')
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
            if len(word) == 1 or idx >= tokenizer.vocab_size:
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
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    if args.perturbed:
        dev_dataset = utils.load_augmented_trigger_dataset(args.dev, templatizer)
    else:
        dev_dataset = utils.load_trigger_dataset(args.dev, templatizer, use_ctx=args.use_ctx)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)
    logger.info('Datasets loaded successfully')
    return train_loader, dev_loader

def find_and_evaluate_triggers(model, tokenizer, templatizer, predictor, embedding_gradient, device, filter, evaluation_fn, train_loader, dev_loader, args):
    logger.info('Initializing trigger tokens')
    # 초기 트리거 토큰 설정
    if args.initial_trigger:
        trigger_ids = tokenizer.convert_tokens_to_ids(args.initial_trigger)
        logger.debug(f'Initial trigger: {args.initial_trigger}')
        logger.debug(f'Trigger ids: {trigger_ids}')
        assert len(trigger_ids) == templatizer.num_trigger_tokens
    else:
        trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
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
            model, predictor, train_loader, averaged_grad, trigger_ids, args, tokenizer, embeddings, filter, token_to_flip
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


def setup_evaluation_function(tokenizer, args, label_map, device):
    if label_map:
        logger.info('Setting up Accuracy evaluation function')
        return AccuracyFn(tokenizer, label_map, device)
    else:
        logger.info('Setting up Loss-based evaluation function')
        return lambda x, y: -get_loss(x, y)

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
    numerator = 0
    denominator = 0
    for model_inputs, labels in tqdm(dev_loader):
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        with torch.no_grad():
            predict_logits = predictor(model_inputs, trigger_ids)
        numerator += evaluation_fn(predict_logits, labels).sum().item()
        denominator += labels.size(0)
    dev_metric = numerator / (denominator + 1e-13)
    logger.info(f'Dev metric: {dev_metric}')
    return dev_metric

def evaluate_candidates(model, predictor, train_loader, averaged_grad, trigger_ids, args, tokenizer, embeddings, filter, token_to_flip):
    """
    Evaluate the candidate tokens to find the best replacement for trigger optimization.
    """
    candidates = hotflip_attack(averaged_grad[token_to_flip],
                                embeddings.weight,
                                increase_loss=False,
                                num_candidates=args.num_cand,
                                filter=filter)

    current_score = 0
    candidate_scores = torch.zeros(args.num_cand, device=trigger_ids.device)
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
            eval_metric = evaluation_fn(predict_logits, labels)

        current_score += eval_metric.sum()
        denom += labels.size(0)

        for i, candidate in enumerate(candidates):
            temp_trigger = trigger_ids.clone()
            temp_trigger[:, token_to_flip] = candidate
            with torch.no_grad():
                predict_logits = predictor(model_inputs, temp_trigger)
                eval_metric = evaluation_fn(predict_logits, labels)
            candidate_scores[i] += eval_metric.sum()

    best_candidate_score = candidate_scores.max()
    best_candidate_idx = candidate_scores.argmax()

    return best_candidate_score, best_candidate_idx, candidates

def run_model(args):

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Loading model, tokenizer, etc.')

    # AutoPopsicle을 사용하여 수정된 BERT 모델 로드
    config = AutoConfig.from_pretrained(args.model_name, num_labels=15)  # 특성 수와 레이블에 맞게 설정
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoPopsicle.from_pretrained(args.model_name, config=config)  # AutoPopsicle 로드
    model.to(device)
    
    embeddings = get_embeddings(model, config)
    embedding_gradient = GradientStorage(embeddings)
    predictor = PredictWrapper(model)

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
        label_field=args.label_field,
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


    # 트리거 탐색 및 평가 함수 호출
    best_trigger_ids, best_dev_metric = find_and_evaluate_triggers(
        model, tokenizer, templatizer, predictor, embedding_gradient, device, filter, evaluation_fn, train_loader, dev_loader, args, embeddings
    )


    # 결과 출력
    best_trigger_tokens = tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))
    logger.info(f'Best tokens: {best_trigger_tokens}')
    logger.info(f'Best dev metric: {best_dev_metric}')

    if args.print_lama:
        print_lama_template(best_trigger_ids, tokenizer, templatizer, args)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True, help='Train data path')
    parser.add_argument('--dev', type=Path, required=True, help='Dev data path')
    parser.add_argument('--template', type=str, help='Template string')
    parser.add_argument('--label-map', type=str, default=None, help='JSON object defining label map')

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
    parser.add_argument('--label-field', type=str, default='label',
                        help='Name of the label field')

    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--eval-size', type=int, default=256, help='Eval size')
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--accumulation-steps', type=int, default=10)
    parser.add_argument('--model-name', type=str, default='bert-base-cased',
                        help='Model name passed to HuggingFace AutoX classes.')
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
    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    run_model(args)
