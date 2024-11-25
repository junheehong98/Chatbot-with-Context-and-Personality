from app.prompt import SYSTEM_PROMPT_TEMPLATE, SITUATION_LIST

# 성격 조합 ID 찾기
def get_personality_id(selected_traits):
    for p in PERSONALITY_LIST:
        if all(trait in p["description"] for trait in selected_traits):
            return p["id"]
    return None

# 시스템 프롬프트 생성
def generate_system_prompt(situation_id):
    situation = SITUATION_LIST[situation_id]

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        situation_description=situation["description"],
        additional_situation_instructions=situation["instructions"]
    )
    return system_prompt

MAX_HISTORY_LENGTH = 5  # 유지할 최대 대화 기록 수

def trim_chat_history(session):
    if len(session['trim_history']) > MAX_HISTORY_LENGTH:
        session['trim_history'] = session['trim_history'][-MAX_HISTORY_LENGTH:]