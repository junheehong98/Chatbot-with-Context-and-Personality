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

    def hook(self, module, grad_in, grad_output):
        # grad_output[0]의 크기는 (batch_size, embedding_dim)
        self._stored_gradient = grad_output[0]

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
        model_inputs.pop('predict_mask', None)  # Remove 'predict_mask' if it exists
       
        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        logits, *_ = self._model(**model_inputs)
       
        return logits

def create_label_combinations():
    """
    2^5(32) 가지의 레이블 조합을 생성합니다. 각 레이블은 0 또는 1의 값을 가질 수 있습니다.
    """
    label_values = [0, 1]
    return list(itertools.product(label_values, repeat=5))

def accuracy_fn(predict_logits, label_ids):
    # 예측값 계산 (로짓이 0보다 크면 1, 아니면 0)
    preds = (predict_logits > 0).long()
    # 각 샘플에 대한 정확도 계산
    correct = (preds == label_ids).float()
    # 전체 정확도 계산 (샘플 및 레이블별 평균)
    acc = correct.mean()
    return acc



def load_pretrained(args, device):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    # 학습된 모델 로드
    config = AutoConfig.from_pretrained(str(args.ckpt_dir))  # Checkpoint directory에서 config 로드
    tokenizer = AutoTokenizer.from_pretrained(str(args.ckpt_dir))  # Checkpoint directory에서 tokenizer 로드
    utils.add_task_specific_tokens(tokenizer)  # 특수 토큰 추가
    model = AutoPopsicle.from_pretrained(str(args.ckpt_dir), config=config)  # 학습된 모델 로드
    model.resize_token_embeddings(len(tokenizer))  # 임베딩 크기를 토크나이저에 맞게 조정
    model.to(device)  # 모델을 지정된 device로 이동
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

def save_best_trigger_tokens(trigger_tokens, scores, combination, output_dir='trigger_tokens'):
    """
    각 조합의 최적 트리거 토큰을 JSON 파일로 저장합니다.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)  # 디렉토리 생성
    file_name = '_'.join(map(str, combination))
    file_path = Path(output_dir) / f'best_trigger_tokens_{file_name}.json'

    # JSON 직렬화를 위해 데이터 타입 변환
    trigger_tokens_serializable = [str(token) for token in trigger_tokens] if not isinstance(trigger_tokens, list) else trigger_tokens


    data = {
        'best_trigger_tokens': trigger_tokens_serializable,
        'best_score': float(scores)  # scores를 float로 변환
    }
    with open(file_path, 'w') as f:
        json.dump(data, f)

    logger.info(f'Trigger tokens saved for combination {combination} to {file_path}')





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
   
    loss_fct = torch.nn.BCEWithLogitsLoss()
    loss = loss_fct(predict_logits, label_ids.float())
    return loss



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


def run_model(args):

    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Loading model, tokenizer, etc.')
    config, model, tokenizer = load_pretrained(args, device)  # 수정된 함수 호출
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

    # 성격 유형 조합 생성
    label_combinations = create_label_combinations()
    start_index = args.start_index if args.start_index < len(label_combinations) else 0
    combinations_to_process = label_combinations[start_index:]

    logger.info(f'Starting from index {start_index} out of {len(label_combinations)} combinations.')

    # 데이터셋 로드
    logger.info('Loading datasets')
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = utils.load_trigger_dataset(args.train, templatizer, use_ctx=args.use_ctx, limit=args.limit)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    dev_dataset = utils.load_trigger_dataset(args.dev, templatizer, use_ctx=args.use_ctx)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

 

    # NOTE: Accuracy can only be computed if a fixed pool of labels is given, which currently
    # requires the label map to be specified. Since producing a label map may be cumbersome (e.g.,
    # for link prediction tasks), we just use (negative) loss as the evaluation metric in these cases.
  

    evaluation_fn = accuracy_fn

    
    # To "filter" unwanted trigger tokens, we subtract a huge number from their logits.
    # filter = torch.zeros(tokenizer.vocab_size, dtype=torch.float32, device=device)
    filter = torch.zeros(embeddings.weight.size(0), dtype=torch.float32, device=device)  # 크기 수정
    if args.filter:
        logger.info('Filtering label tokens.')
        if label_map:
            for label_tokens in label_map.values():
                label_ids = utils.encode_label(tokenizer, label_tokens).unsqueeze(0)
                filter[label_ids] = -1e32
        else:
            for _, label_ids in train_dataset:
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



    # ** 각 조합에 대해 트리거 탐색 수행 **
    for combination in tqdm(combinations_to_process, desc="Processing label combinations"):
        logger.info(f'Evaluating combination: {combination}')

        # 초기 트리거 토큰 설정
        trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
        trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)
        best_trigger_ids = trigger_ids.clone()
        best_dev_metric = -float('inf')   


        for i in range(args.iters):

            logger.info(f'Iteration: {i}')

            logger.info('Accumulating Gradient')
            model.zero_grad()

            pbar = tqdm(range(args.accumulation_steps))
            train_iter = iter(train_loader)
            averaged_grad = None

            # Accumulate
            for step in pbar:

                # Shuttle inputs to GPU
                try:
                    model_inputs, labels = next(train_iter)
                except:
                    logger.warning(
                        'Insufficient data for number of accumulation steps. '
                        'Effective batch size will be smaller than specified.'
                    )
                    break
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                predict_logits = predictor(model_inputs, trigger_ids)
                loss = get_loss(predict_logits, labels)
                loss.backward()

                grad = embedding_gradient.get()
                bsz, _, emb_dim = grad.size()
                selection_mask = model_inputs['trigger_mask'].unsqueeze(-1)
                grad = torch.masked_select(grad, selection_mask)
                grad = grad.view(bsz, templatizer.num_trigger_tokens, emb_dim)

                if averaged_grad is None:
                    averaged_grad = grad.sum(dim=0) / args.accumulation_steps
                else:
                    averaged_grad += grad.sum(dim=0) / args.accumulation_steps

            # Evaluate candidates

            token_to_flip = random.randrange(templatizer.num_trigger_tokens)
            candidates = hotflip_attack(averaged_grad[token_to_flip],
                                        embeddings.weight,
                                        increase_loss=False,
                                        num_candidates=args.num_cand,
                                        filter=filter)

            current_score = 0
            candidate_scores = torch.zeros(args.num_cand, device=device)
            for idx, candidate in enumerate(candidates):
                    temp_trigger = trigger_ids.clone()
                    temp_trigger[:, token_to_flip] = candidate
                    with torch.no_grad():
                        predict_logits = predictor({k: v.to(device) for k, v in model_inputs.items()}, temp_trigger)
                        acc = accuracy_fn(predict_logits, labels)
                    candidate_scores[idx] += acc.item() * labels.numel()
            best_candidate_score = candidate_scores.max()
            best_candidate_idx = candidate_scores.argmax()
            if best_candidate_score > best_dev_metric:
                trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
                best_trigger_ids = trigger_ids.clone()
                best_dev_metric = best_candidate_score
        # 결과 저장
        best_trigger_tokens = tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))
        logger.info(f'Best tokens for combination {combination}: {best_trigger_tokens}')
        logger.info(f'Best dev metric for combination {combination}: {best_dev_metric}')
        save_best_trigger_tokens(best_trigger_tokens, best_dev_metric, combination, output_dir='trigger_tokens')

    logger.info('All combinations processed successfully.')









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

    parser.add_argument('--initial-trigger', nargs='+', type=str, default=["[CLS]", "[CLS]", "[CLS]"], help='Manual prompt')
    parser.add_argument('--label-field', nargs='+',type=str, default=['Label1', 'Label2', 'Label3', 'Label4', 'Label5'],
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
    parser.add_argument('--start-index', type=int, default=0,
                    help='Start index for processing label combinations')

    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    run_model(args)