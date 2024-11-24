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
from tqdm import tqdm, trange


from autoprompt.popsicle import AutoPopsicle
import autoprompt.utils as utils


###


# test.py에서 로그 보고 싶을때

# create_trigger_user.py 파일 상단
logging.basicConfig(
    level=logging.INFO,  # 로그 레벨 설정
    format='%(asctime)s - %(levelname)s - %(message)s',  # 로그 포맷
    handlers=[logging.StreamHandler()]  # 로그를 콘솔에 출력
)



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
    # acc = correct.sum() / correct.numel()  # 정확도: 맞은 개수 / 전체 개수


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

    # Ensure label_ids matches predict_logits shape
    if label_ids.dim() == 1:  # If label_ids is [5], expand to [1, 5]
        label_ids = label_ids.unsqueeze(0)
   
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


def run_model(user_prompt, personality, ckpt_dir='ckpt/', num_labels=5):


    logger = logging.getLogger(__name__)

    # 기본 설정
    template = "<s> {input_text} [T] [T] [T] [P] . </s>"
    num_cand = 100
    accumulation_steps = 30
    bsz = 1
    eval_size = 48
    iters = 5
    seed = 42

    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   
    set_seed(seed)

    # 모델, 토크나이저 로드
    logger.info('Loading model and tokenizer...')
    config = AutoConfig.from_pretrained(ckpt_dir)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    utils.add_task_specific_tokens(tokenizer)
    model = AutoPopsicle.from_pretrained(ckpt_dir, config=config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    embeddings = get_embeddings(model, config)
    embedding_gradient = GradientStorage(embeddings)
    predictor = PredictWrapper(model)    

    # 템플라이저 설정
    templatizer = utils.TriggerTemplatizer(
        template=template,
        config=config,
        tokenizer=tokenizer,
        label_map=None,
        label_field=["Label1", "Label2", "Label3", "Label4", "Label5"],
        tokenize_labels=True,
        add_special_tokens=False,
        use_ctx=False,
    )

    # 데이터 생성
    data_instance = {
        "input_text": user_prompt,
        "Label1": personality[0],
        "Label2": personality[1],
        "Label3": personality[2],
        "Label4": personality[3],
        "Label5": personality[4],
    }


    model_inputs, labels = templatizer(data_instance)
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    labels = labels.to(device)
    

    # 필터 생성
    logger.info("Creating filter...")
    filter = torch.zeros(embeddings.weight.size(0), dtype=torch.float32, device=device)
    for word, idx in tokenizer.get_vocab().items():
        if len(word) == 1 or idx in tokenizer.all_special_ids or idx >= len(tokenizer):
            filter[idx] = -1e32


    # 트리거 탐색
    logger.info("Searching for optimal triggers...")

    # 트리거 초기화
    trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0) 
    best_trigger_ids = trigger_ids.clone()
    best_dev_metric = -float("inf")    


    # 탐색 시작
    logger.info("Starting trigger optimization...")
    for i in range(iters):
        logger.info(f"Iteration: {i+1}/{iters}")  # 현재 반복 횟수 출력
        logger.info(f"Current trigger tokens: {tokenizer.convert_ids_to_tokens(trigger_ids.squeeze(0))}")
        
        model.zero_grad()


        predict_logits = predictor(model_inputs, trigger_ids)
        loss = get_loss(predict_logits, labels)
        loss.backward()                  
            

        grad = embedding_gradient.get()
        selection_mask = model_inputs["trigger_mask"].unsqueeze(-1)
        grad = torch.masked_select(grad, selection_mask)
        grad = grad.view(templatizer.num_trigger_tokens, embeddings.weight.size(1))

        # Evaluate candidates

        token_to_flip = random.randrange(templatizer.num_trigger_tokens)
        candidates = hotflip_attack(
            grad[token_to_flip],
            embeddings.weight,
            increase_loss=False,
            num_candidates=num_cand,
            filter=filter,
        )

        

        candidate_scores = torch.zeros(len(candidates), device=device)
        for idx, candidate in enumerate(candidates):
            temp_trigger_ids = trigger_ids.clone()
            temp_trigger_ids[:, token_to_flip] = candidate
            with torch.no_grad():
                predict_logits = predictor(model_inputs, temp_trigger_ids)
                acc = accuracy_fn(predict_logits, labels)
            candidate_scores[idx] += acc.item() * labels.numel()

        best_candidate_score = candidate_scores.max()
        best_candidate_idx = candidate_scores.argmax()

        if best_candidate_score > best_dev_metric:
            trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
            best_trigger_ids = trigger_ids.clone()
            best_dev_metric = best_candidate_score

        logger.info(f"Best candidate score: {best_candidate_score}")
        logger.info(f"Best trigger tokens so far: {tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))}")
        logger.info(f"Best development metric so far: {best_dev_metric.item()}")


    # 최적의 트리거를 토큰으로 변환
    best_trigger_tokens = tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))
    

    # 최종 프롬프트 생성
    final_prompt = template.replace("{input_text}", user_prompt)
    for token in best_trigger_tokens:
        final_prompt = final_prompt.replace("[T]", token, 1)


    logger.info(f"Best trigger tokens: {best_trigger_tokens}")
    logger.info(f"Final prompt: {final_prompt}")
    # 최적 성능 메트릭 출력 시
    logger.info(f"Best development metric so far: {best_dev_metric.item()}")

    # 결과 반환
    return {
        "best_trigger_tokens": best_trigger_tokens,
        "final_prompt": final_prompt,
        "best_dev_metric": best_dev_metric.item(),  # 텐서를 숫자로 변환
    }





    



















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