import csv
import copy
import json
import logging
from multiprocessing.sharedctypes import Value
import random
from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence


MAX_CONTEXT_LEN = 50


logger = logging.getLogger(__name__)


def pad_squeeze_sequence(sequence, *args, **kwargs):
    """Squeezes fake batch dimension added by tokenizer before padding sequence."""
    return pad_sequence([x.squeeze(0) for x in sequence], *args, **kwargs)


class OutputStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_output = None
        module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self._stored_output = output

    def get(self):
        return self._stored_output


class ExponentialMovingAverage:
    def __init__(self, weight=0.3):
        self._weight = weight
        self.reset()

    def update(self, x):
        self._x += x
        self._i += 1

    def reset(self):
        self._x = 0
        self._i = 0

    def get_metric(self):
        return self._x  / (self._i + 1e-13)


class Collator:
    """
    Collates transformer outputs.
    """
    def __init__(self, pad_token_id=0):
        self._pad_token_id = pad_token_id

    def __call__(self, features):
        # Separate the list of inputs and labels
        model_inputs, labels = list(zip(*features))
        # Assume that all inputs have the same keys as the first
        proto_input = model_inputs[0]
        keys = list(proto_input.keys())
        padded_inputs = {}
        for key in keys:
            if key == 'input_ids':
                padding_value = self._pad_token_id
            else:
                padding_value = 0
            # NOTE: We need to squeeze to get rid of fake batch dim.
            sequence = [x[key] for x in model_inputs]
            padded = pad_squeeze_sequence(sequence, batch_first=True, padding_value=padding_value)
            padded_inputs[key] = padded
        # 레이블을 직접 스택
        labels = torch.stack(labels)
        # labels = pad_squeeze_sequence(labels, batch_first=True, padding_value=0)
        return padded_inputs, labels


def encode_label(tokenizer, label, tokenize=False):
    """
    Helper function for encoding labels. Deals with the subtleties of handling multiple tokens.
    """
    if isinstance(label, str):
        if tokenize:
            # Ensure label is properly tokenized, and only retain first token
            # if it gets split into multiple tokens. TODO: Make sure this is
            # desired behavior.
            tokens = tokenizer.tokenize(label)
            if len(tokens) > 1:
                raise ValueError(f'Label "{label}" gets mapped to multiple tokens.')
            if tokens[0] == tokenizer.unk_token:
                raise ValueError(f'Label "{label}" gets mapped to unk.')
            label = tokens[0]
        encoded = torch.tensor(tokenizer.convert_tokens_to_ids([label])).unsqueeze(0)
    elif isinstance(label, list):
        encoded = torch.tensor(tokenizer.convert_tokens_to_ids(label)).unsqueeze(0)
    elif isinstance(label, int):
        encoded = torch.tensor([[label]])
    return encoded


class TriggerTemplatizer:
    """
    An object to facilitate creating transformers-friendly triggers inputs from a template.

    Parameters
    ==========
    template : str
        The template string, comprised of the following tokens:
            [T] to mark a trigger placeholder.
            [P] to mark a prediction placeholder.
            {fields} arbitrary fields instantiated from the dataset instances.
        For example a NLI template might look like:
            "[T] [T] [T] {premise} [P] {hypothesis}"
    tokenizer : PretrainedTokenizer
        A HuggingFace tokenizer. Must have special trigger and predict tokens.
    add_special_tokens : bool
        Whether or not to add special tokens when encoding. Default: False.
    """
    def __init__(self,
                 template,
                 config,
                 tokenizer,
                 label_field='label',
                 label_map=None,
                 tokenize_labels=False,
                 add_special_tokens=False,
                 use_ctx=False):
        if not hasattr(tokenizer, 'predict_token') or \
           not hasattr(tokenizer, 'trigger_token'):
            raise ValueError(
                'Tokenizer missing special trigger and predict tokens in vocab.'
                'Use `utils.add_special_tokens` to add them.'
            )
        self._template = template
        self._config = config
        self._tokenizer = tokenizer
        self._label_field = label_field
        self._label_map = label_map
        self._tokenize_labels = tokenize_labels
        self._add_special_tokens = add_special_tokens
        self._use_ctx = use_ctx

    @property
    def num_trigger_tokens(self):
        return sum(token == '[T]' for token in self._template.split())

    def __call__(self, format_kwargs):
        # Format the template string
        format_kwargs = format_kwargs.copy()
        # 템플릿 생성
        text = format_kwargs['text']
        labels = [format_kwargs[f'Label{i}'] for i in range(1, 6)]       
    
        
        if text is None:
            raise Exception(f'Bad data: {text}')
        

        # Have the tokenizer encode the text and process the output to:
        # - Create a trigger and predict mask
        # - Replace the predict token with a mask token
        model_inputs = self._tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            return_tensors='pt'
        )


        input_ids = model_inputs['input_ids']
        trigger_mask = input_ids.eq(self._tokenizer.trigger_token_id)
        predict_mask = input_ids.eq(self._tokenizer.predict_token_id)

        input_ids[predict_mask] = self._tokenizer.mask_token_id

        model_inputs['trigger_mask'] = trigger_mask
        model_inputs['predict_mask'] = predict_mask

        # For relation extraction with BERT, update token_type_ids to reflect the two different sequences
        if self._use_ctx and self._config.model_type == 'bert':
            sep_token_indices = (input_ids.squeeze(0) == self._tokenizer.convert_tokens_to_ids(self._tokenizer.sep_token)).nonzero().flatten()
            sequence_b_indices = torch.arange(sep_token_indices[0], sep_token_indices[1] + 1).long().unsqueeze(0)
            model_inputs['token_type_ids'].scatter_(1, sequence_b_indices, 1)

       
            
        # 레이블을 텐서로 변환 (숫자 그대로 사용)
        label_ids = torch.tensor(labels, dtype=torch.long)  # 다중 레이블을 하나의 텐서로 결합
    
        return model_inputs, label_ids


def add_task_specific_tokens(tokenizer):
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[T]', '[P]', '[Y]']
    })
    tokenizer.trigger_token = '[T]'
    tokenizer.trigger_token_id = tokenizer.convert_tokens_to_ids('[T]')
    tokenizer.predict_token = '[P]'
    tokenizer.predict_token_id = tokenizer.convert_tokens_to_ids('[P]')
    # NOTE: BERT and RoBERTa tokenizers work properly if [X] is not a special token...
    # tokenizer.lama_x = '[X]'
    # tokenizer.lama_x_id = tokenizer.convert_tokens_to_ids('[X]')
    tokenizer.lama_y = '[Y]'
    tokenizer.lama_x_id = tokenizer.convert_tokens_to_ids('[Y]')

def load_csv(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def load_tsv(fname):
    with open(fname, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            yield row


def load_jsonl(fname):
    with open(fname, 'r') as f:
        for line in f:
            yield json.loads(line)


LOADERS = {
    '.tsv': load_tsv,
    '.jsonl': load_jsonl,
    '.csv': load_csv  # CSV 로더 추가

}


def load_trigger_dataset(fname, templatizer, use_ctx, limit=None):
    loader = LOADERS[fname.suffix]
    instances = []

    for x in loader(fname):       

        # logger.info(f"Loaded row: {x}")  # 디버깅 추가

        try:

            # 필드 유효성 검사
            if not all(field in x for field in ['text', 'Label1', 'Label2', 'Label3', 'Label4', 'Label5']):
                logger.warning(f"Skipping row due to missing fields: {x}")
                continue

            # input_text = x['text']
            # labels = [int(x[f'Label{i}']) for i in range(1, 6)]
            # 레이블을 문자열로 추출 후 유효성 검사
            model_inputs, label_ids = templatizer({
                'text': x['text'],  # `row`를 `x`로 변경
                'Label1': int(x['Label1']),
                'Label2': int(x['Label2']),
                'Label3': int(x['Label3']),
                'Label4': int(x['Label4']),
                'Label5': int(x['Label5']),
            })
            instances.append((model_inputs, label_ids))
        except ValueError as e:
            logger.warning(f"Skipping row due to error: {e}")
            continue  # 에러 발생 시 해당 행 건너뜀
    logger.info(f"Number of instances loaded: {len(instances)}")  # 디버깅 추가
    # 샘플 제한이 설정된 경우 제한된 데이터 반환
    if limit:
        instances = random.sample(instances, min(limit, len(instances)))

    return instances
    '''

    for x in loader(fname):
        try:
            if use_ctx:
                # For relation extraction, skip facts that don't have context sentence
                if 'evidences' not in x:
                    logger.warning('Skipping RE sample because it lacks context sentences: {}'.format(x))
                    continue

                evidences = x['evidences']
                    
                # Randomly pick a context sentence
                obj_surface, masked_sent = random.choice([(evidence['obj_surface'], evidence['masked_sentence']) for evidence in evidences])
                words = masked_sent.split()
                if len(words) > MAX_CONTEXT_LEN:
                    # If the masked sentence is too long, use the first X tokens. For training we want to keep as many samples as we can.
                    masked_sent = ' '.join(words[:MAX_CONTEXT_LEN])
                
                # If truncated context sentence still has MASK, we need to replace it with object surface
                # We explicitly use [MASK] because all TREx fact's context sentences use it
                context = masked_sent.replace('[MASK]', obj_surface)
                x['context'] = context
                model_inputs, label_id = templatizer(x)
            else:

                input_text = x['text']  # Use 'text' column for input
                labels = [int(x[f'Label{i}']) for i in range(1, 6)]  # Read Label1 to Label5

                # model_inputs, label_id = templatizer(x)

                # Create model inputs using templatizer
                model_inputs, label_id = templatizer({
                    'input_text': input_text,
                    'Label1': labels[0],
                    'Label2': labels[1],
                    'Label3': labels[2],
                    'Label4': labels[3],
                    'Label5': labels[4],
                })






        except ValueError as e:
            logger.warning('Encountered error "%s" when processing "%s".  Skipping.', e, x)
            continue
        else:
            instances.append((model_inputs, label_id))
    if limit:
        return random.sample(instances, limit)
    else:
        return instances
        '''


def load_augmented_trigger_dataset(fname, templatizer, limit=None):
    loader = LOADERS[fname.suffix]
    instances = []

    # For augmented relation extraction, we need to replace obj_label with another obj_label, and replace obj_surface with a surface form of the new obj_label
    unique_objs_dict = defaultdict(list)
    # Also for augmented relation extraction, we need to accumulate all facts and process them afterwards
    facts = []

    for x in loader(fname):
        try:
            if 'sub_label' in x and 'obj_label' in x:
                sub_label = x['sub_label']
                obj_label = x['obj_label']

                # For relation extraction, skip facts that don't have context sentence
                if 'evidences' not in x:
                    logger.warning('Skipping RE sample because it lacks context sentences: {}'.format(x))
                    continue

                evidences = x['evidences']

                # Gather all UNIQUE objects and their surface forms if its augmented relation extraction
                for evidence in evidences:
                    obj_surface = evidence['obj_surface']
                    masked_sent = evidence['masked_sentence']
                    unique_objs_dict[obj_label].append(obj_surface)
                    
                # Randomly pick a context sentence
                obj_surface, masked_sent = random.choice([(evidence['obj_surface'], evidence['masked_sentence']) for evidence in evidences])
                words = masked_sent.split()
                if len(words) > MAX_CONTEXT_LEN:
                    # If the masked sentence is too long, use the first X tokens. For training we want to keep as many samples as we can.
                    masked_sent = ' '.join(words[:MAX_CONTEXT_LEN])
            
                x['context'] = masked_sent
                facts.append(x)
            else:
                # sub_label과 obj_label이 없는 경우, input_text와 label들을 사용하여 처리
                input_text = x['input_text']
                label1 = int(x['label1'])
                label2 = int(x['label2'])
                label3 = int(x['label3'])
                label4 = int(x['label4'])
                label5 = int(x['label5'])
                labels = [label1, label2, label3, label4, label5]

                # 템플릿과 레이블을 이용해 데이터 생성
                model_inputs, label_id = templatizer({
                    'input_text': input_text,
                    'label1': label1,
                    'label2': label2,
                    'label3': label3,
                    'label4': label4,
                    'label5': label5,
                })
                instances.append((model_inputs, label_id))
                
        except ValueError as e:
            logger.warning('Encountered error "%s" when processing "%s".  Skipping.', e, x)

    # Go through all facts and replace each object with a new one. Also insert the new object (surface form) into the masked sentence
    synth_facts = []
    for fact in facts:
        sub_label = fact['sub_label']
        obj_label = fact['obj_label']
        masked_sent = fact['context']
        # print('Original fact: ({}, {}, {})'.format(sub_label, obj_label, masked_sent))
        synth_obj_label = random.choice([x for x in unique_objs_dict.keys() if x != obj_label])
        synth_obj_surface = random.choice(unique_objs_dict[synth_obj_label])
        synth_ctx = masked_sent.replace('[MASK]', synth_obj_surface)
        # print('Synthetic fact: ({}, {}, {})\n'.format(sub_label, synth_obj_label, synth_ctx))
        # Reassign the labels and context sentence
        synth_fact = copy.deepcopy(fact)
        synth_fact['sub_label'] = sub_label
        synth_fact['obj_label'] = synth_obj_label
        synth_fact['context'] = synth_ctx
        synth_facts.append(synth_fact)

    # Go through facts, templatize each one, then append them to instances
    for fact in synth_facts:
        try:
            model_inputs, label_id = templatizer(fact)
            instances.append((model_inputs, label_id))
        except ValueError as e:
            print(e)

    if limit:
        return random.sample(instances, limit)
    else:
        return instances


def load_classification_dataset(
    fname,
    tokenizer,
    input_field_a,
    input_field_b=None,
    label_field=['Label1', 'Label2', 'Label3', 'Label4', 'Label5'],
    label_map=None,
    limit=None
):
    """
    Loads a dataset for classification

    Parameters
    ==========
    tokenizer : transformers.PretrainedTokenizer
        Maps text to id tensors.
    sentence1 :
    """
    instances = []    
    loader = LOADERS[fname.suffix]
    for instance in loader(fname):
        logger.debug(instance)
        model_inputs = tokenizer.encode_plus(
            instance[input_field_a],
            instance[input_field_b] if input_field_b else None,
            add_special_tokens=True,
            # add_prefix_space=True,
            return_tensors='pt'
        )
        logger.debug(model_inputs)
        
        
        # print(f"Loaded instance: {instance}")  # 디버깅 출력
        # label = instance[label_field]
        # print(f"Loaded label: {label}")  # 디버깅 출력
        # if label not in label_map:
        #     label_map[label] = len(label_map)
        # label_id = label_map[label]
        # label_id = torch.tensor([[label_id]])  # To make collator expectation
        # logger.debug(f'Label id: {label_id}')
        # instances.append((model_inputs, label_id))
        
        try:
            # print(f"Loaded instance: {instance}")  # 로드된 데이터 인스턴스를 출력
            # 각 레이블 컬럼에서 레이블 값을 추출하여 리스트로 저장
            labels = [int(instance[label_name]) for label_name in label_field]
            # print(f"Loaded label: {label}")  # 로드된 레이블을 출력         
           
            
            
            # 레이블을 tensor로 변환
            label_id = torch.tensor(labels, dtype=torch.long)  # (5,) 형태로 변환
            logger.debug(f'Label id: {label_id}')
            instances.append((model_inputs, label_id))
        except Exception as e:
            logger.error(f"Error processing instance: {instance}, Error: {e}")
            
            
    if limit:
        instances = random.sample(instances, limit)
    return instances, label_map
