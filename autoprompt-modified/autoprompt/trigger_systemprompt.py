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




