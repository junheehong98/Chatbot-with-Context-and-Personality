import streamlit as st
import openai
from openai import OpenAI

# OpenAI API Key 설정 
openai.api_key ="Key"
client = OpenAI( 
    # This is the default and can be omitted
    api_key="secret_key"
)
# 상황 리스트 정의
SITUATIONS = ["Subway", "Lunch", "Library", "Cafe", "Class Break"]

# 성격 특성 5가지 선택 목록
characteristics = [
    ["Negative", "Positive"],
    ["Aggressive", "Peaceful"],
    ["Conservative", "Open-minded"],
    ["Introverted", "Extroverted"],
    ["Unstable", "Confident"]
]

# 기본 시스템 프롬프트 템플릿
SYSTEM_PROMPT_TEMPLATE = """
You are a small talk chatbot. Engage in a conversation that fits the given context and personality.
Situation: {situation_description}
Personality: {personality_description}

# Guidelines
1. Maintain a consistent tone and style according to the specified personality.
2. Focus on making the conversation enjoyable and contextually appropriate.
3. Avoid controversial or sensitive topics unless specifically requested.
4. If the conversation becomes awkward, use humor or empathy to lighten the mood.

# Situation Details:
{additional_situation_instructions}

# Personality Traits:
{additional_personality_instructions}
"""

# 상황과 성격의 정의
SITUATION_LIST = [
    {"id": 0, "description": "Subway", "instructions": "You're on a crowded subway. Try to make light conversation without intruding on personal space."},
    {"id": 1, "description": "Lunch", "instructions": "You're having lunch with a friend or classmate. Discuss food preferences, recent events, or light-hearted topics."},
    {"id": 2, "description": "Library", "instructions": "You're studying in the library. Keep the conversation quiet and respectful, focusing on study tips or university life."},
    {"id": 3, "description": "Cafe", "instructions": "You're sitting in a cafe, enjoying a drink. Talk about favorite beverages, recent trends, or what brings you to the cafe."},
    {"id": 4, "description": "Class Break", "instructions": "You're on a break between classes. Talk about classes, assignments, or make casual plans for later."}
]

PERSONALITY_LIST = [
    {"id": 0, "description": "Negative, Aggressive, Conservative, Introverted, Unstable", "instructions": "Be critical, assertive, and reserved. You tend to view things negatively and are not easily persuaded by new ideas. You prefer to keep to yourself and often feel unstable in social situations."},
    {"id": 1, "description": "Negative, Aggressive, Conservative, Introverted, Confident", "instructions": ""},
    {"id": 2, "description": "Negative, Aggressive, Conservative, Extroverted, Unstable", "instructions": ""},
    {"id": 3, "description": "Negative, Aggressive, Conservative, Extroverted, Confident", "instructions": ""},
    {"id": 4, "description": "Negative, Aggressive, Open-minded, Introverted, Unstable", "instructions": ""},
    {"id": 5, "description": "Negative, Aggressive, Open-minded, Introverted, Confident", "instructions": ""},
    {"id": 6, "description": "Negative, Aggressive, Open-minded, Extroverted, Unstable", "instructions": ""},
    {"id": 7, "description": "Negative, Aggressive, Open-minded, Extroverted, Confident", "instructions": ""},
    {"id": 8, "description": "Negative, Peaceful, Conservative, Introverted, Unstable", "instructions": ""},
    {"id": 9, "description": "Negative, Peaceful, Conservative, Introverted, Confident", "instructions": ""},
    {"id": 10, "description": "Negative, Peaceful, Conservative, Extroverted, Unstable", "instructions": ""},
    {"id": 11, "description": "Negative, Peaceful, Conservative, Extroverted, Confident", "instructions": ""},
    {"id": 12, "description": "Negative, Peaceful, Open-minded, Introverted, Unstable", "instructions": ""},
    {"id": 13, "description": "Negative, Peaceful, Open-minded, Introverted, Confident", "instructions": ""},
    {"id": 14, "description": "Negative, Peaceful, Open-minded, Extroverted, Unstable", "instructions": ""},
    {"id": 15, "description": "Negative, Peaceful, Open-minded, Extroverted, Confident", "instructions": ""},
    {"id": 16, "description": "Positive, Aggressive, Conservative, Introverted, Unstable", "instructions": ""},
    {"id": 17, "description": "Positive, Aggressive, Conservative, Introverted, Confident", "instructions": ""},
    {"id": 18, "description": "Positive, Aggressive, Conservative, Extroverted, Unstable", "instructions": ""},
    {"id": 19, "description": "Positive, Aggressive, Conservative, Extroverted, Confident", "instructions": ""},
    {"id": 20, "description": "Positive, Aggressive, Open-minded, Introverted, Unstable", "instructions": ""},
    {"id": 21, "description": "Positive, Aggressive, Open-minded, Introverted, Confident", "instructions": ""},
    {"id": 22, "description": "Positive, Aggressive, Open-minded, Extroverted, Unstable", "instructions": ""},
    {"id": 23, "description": "Positive, Aggressive, Open-minded, Extroverted, Confident", "instructions": ""},
    {"id": 24, "description": "Positive, Peaceful, Conservative, Introverted, Unstable", "instructions": ""},
    {"id": 25, "description": "Positive, Peaceful, Conservative, Introverted, Confident", "instructions": ""},
    {"id": 26, "description": "Positive, Peaceful, Conservative, Extroverted, Unstable", "instructions": ""},
    {"id": 27, "description": "Positive, Peaceful, Conservative, Extroverted, Confident", "instructions": ""},
    {"id": 28, "description": "Positive, Peaceful, Open-minded, Introverted, Unstable", "instructions": ""},
    {"id": 29, "description": "Positive, Peaceful, Open-minded, Introverted, Confident", "instructions": ""},
    {"id": 30, "description": "Positive, Peaceful, Open-minded, Extroverted, Unstable", "instructions": ""},
    {"id": 31, "description": "Positive, Peaceful, Open-minded, Extroverted, Confident", "instructions": ""}
]

# 성격 조합 ID 찾기
def get_personality_id(selected_traits):
    for p in PERSONALITY_LIST:
        if all(trait in p["description"] for trait in selected_traits):
            return p["id"]
    return None

# 시스템 프롬프트 생성
def generate_system_prompt(situation_id, personality_id):
    situation = SITUATION_LIST[situation_id]
    personality = PERSONALITY_LIST[personality_id]

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        situation_description=situation["description"],
        personality_description=personality["description"],
        additional_situation_instructions=situation["instructions"],
        additional_personality_instructions=personality["instructions"],
    )
    return system_prompt
# Streamlit 앱 설정
st.title("Small Talk Chatbot Service")

if "system_prompt" not in st.session_state:
    # 상황 선택 및 성격 선택 화면
    selected_situation = st.selectbox("Select a Situation:", SITUATIONS)

    selected_characteristics = []
    for i, characteristic in enumerate(characteristics):
        selected_characteristics.append(
            st.selectbox(f"Select {i+1} Personality Trait:", characteristic)
        )

    personality_id = get_personality_id(selected_characteristics)

    if personality_id is not None:
        if st.button("Generate System Prompt"):
            st.session_state.system_prompt = generate_system_prompt(SITUATIONS.index(selected_situation), personality_id)
            st.session_state.chat_history = []  # 초기 대화 기록 설정
            st.experimental_rerun()

else:
    # 챗봇 대화창
    st.subheader("Chat with the Small Talk Chatbot!")
    
    # 대화 기록 출력
    if "chat_history" in st.session_state:
        for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
            st.text(f"You: {user_msg}")
            st.text_area(f"Chatbot: {bot_msg}")
    
    user_input = st.text_input("Your Message:")

    if st.button("Send to Chatbot"):
        if not user_input.strip():
            st.error("Please enter a message before sending.")
        else:
            # 기존 대화 기록과 함께 시스템 프롬프트 전달
            conversation = [{"role": "system", "content": st.session_state.system_prompt}]
            for user_msg, bot_msg in st.session_state.chat_history:
                conversation.append({"role": "user", "content": user_msg})
                conversation.append({"role": "assistant", "content": bot_msg})
            
            # 새로운 사용자 메시지 추가
            conversation.append({"role": "user", "content": user_input})
            
            # OpenAI API 요청
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation,
                max_tokens=300,
                n=1,
                stop=["\n"],
                response_format={"type": "text"}
            )
            
            chatbot_response = response.choices[0].message.content.strip()
            
            # 대화 기록에 추가
            st.session_state.chat_history.append((user_input, chatbot_response))
            
            # 대화 기록을 갱신하여 출력
            st.experimental_rerun()