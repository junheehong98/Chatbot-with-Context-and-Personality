from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import openai
from openai import OpenAI
from prompt import SYSTEM_PROMPT_TEMPLATE, SITUATION_LIST, PERSONALITY_LIST
from utils import get_personality_id, generate_system_prompt, trim_chat_history
import json
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# OpenAI API 키 설정
openai.api_key = ''
client = OpenAI(
    api_key="secret_key"
)

# 상황 리스트
SITUATIONS = ["Subway", "Lunch", "Library", "Cafe", "Class Break"]

# 성격 특성 목록
characteristics = [
    ["Negative", "Positive"],
    ["Aggressive", "Peaceful"],
    ["Conservative", "Open-minded"],
    ["Introverted", "Extroverted"],
    ["Unstable", "Confident"]
]

# 메인 페이지 (성격 및 상황 선택)
@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        # 선택된 상황과 성격 특성을 세션에 저장
        selected_situation = request.form.get('situation')
        selected_characteristics = [request.form.get(f'characteristic{i}') for i in range(5)]
        session['selected_situation'] = selected_situation
        session['selected_characteristics'] = selected_characteristics
        
        return redirect(url_for('chat'))

    return render_template('main.html', situations=SITUATIONS, characteristics=characteristics)

# # 대화 페이지
# @app.route('/chat', methods=['GET', 'POST'])
# def chat():
#     session.permanent = True
#     if 'trim_history' not in session:
#         session['chat_history'] = []
#         session['trim_history'] = []  # 대화 기록 초기화

#     personality_id = get_personality_id(session['selected_characteristics'])
#     system_prompt = generate_system_prompt(SITUATIONS.index(session['selected_situation']), personality_id)

#     if request.method == 'POST':
#         user_input = request.form.get('user_input')
#         if user_input.strip():
#             # OpenAI API 요청
#             conversation = [{"role": "system", "content": system_prompt}]
#             for user_msg, bot_msg in session['trim_history']:
#                 conversation.append({"role": "user", "content": user_msg})
#                 conversation.append({"role": "assistant", "content": bot_msg})
            
#             conversation.append({"role": "user", "content": user_input})

#             response = client.chat.completions.create(
#                 model="gpt-4o-mini",
#                 messages=conversation,
#                 max_tokens=300,
#                 n=1,
#                 stop=["\n"],
#                 response_format={"type": "text"}
#             )

#             bot_response = response.choices[0].message.content.strip()

#             # 대화 기록에 추가
#             session['chat_history'].append((user_input, bot_response))
#             session['trim_history'].append((user_input, bot_response))

#             # 대화 기록 정리 
#             trim_chat_history(session)

#     return render_template('chat.html', chat_history=session['chat_history'])
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    session.permanent = True
    if 'trim_history' not in session:
        session['chat_history'] = []
        session['trim_history'] = []  # 대화 기록 초기화

    personality_id = get_personality_id(session['selected_characteristics'])
    system_prompt = generate_system_prompt(SITUATIONS.index(session['selected_situation']), personality_id)

    if request.method == 'POST':
        user_input = request.json.get('user_input')  # AJAX 요청의 JSON 데이터에서 user_input 가져옴
        if user_input and user_input.strip():  # 유효한 입력인지 확인
            # OpenAI API 요청
            conversation = [{"role": "system", "content": system_prompt}]
            for user_msg, bot_msg in session['trim_history']:
                conversation.append({"role": "user", "content": user_msg})
                conversation.append({"role": "assistant", "content": bot_msg})

            conversation.append({"role": "user", "content": user_input})

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversation,
                max_tokens=300,
                n=1,
                stop=["\n"],
                response_format={"type": "text"}
            )

            bot_response = response.choices[0].message.content.strip()

            # 대화 기록에 추가
            session['chat_history'].append((user_input, bot_response))
            session['trim_history'].append((user_input, bot_response))

            # 대화 기록 정리 (사용자 정의 함수)
            trim_chat_history(session)

            # 새로운 챗봇 응답을 JSON으로 반환
            return jsonify({"bot_msg": bot_response})

    # GET 요청 처리
    return render_template('chat.html', chat_history=session['chat_history']) 


# 유저 평가를 저장할 딕셔너리
ratings_dict = {}

# 평가 처리
@app.route('/evaluate', methods=['POST'])
def evaluate():
    bot_msg = request.form.get('bot_msg')
    rating = request.form.get('rating')
    
    # 현재 세션에서 선택된 상황과 성격 정보 가져오기
    selected_situation = session.get('selected_situation')
    selected_characteristics = session.get('selected_characteristics')
    
    # key는 상황과 성격 조합, value는 [챗봇 응답, 유저 평가]
    key = (selected_situation, tuple(selected_characteristics))
    
    # 기존 값을 덮어쓰지 않고 append
    if key not in ratings_dict:
        ratings_dict[key] = []
    
    ratings_dict[key].append([bot_msg, rating])
    
    # chat_history 그대로 유지한 채로 chat 페이지 렌더링
    return render_template('chat.html', chat_history=session['chat_history'])

# 유저가 대화 종료 시 데이터 저장
@app.route('/end_chat', methods=['POST'])
def end_chat():
    user_id = session.get('user_id', 'unknown_user')  # 유저 ID 또는 식별 정보 가져오기
    file_path = f'chat_ratings_{user_id}.json'
    
    # dictionary를 JSON 파일로 저장
    with open(file_path, 'w') as f:
        json.dump(ratings_dict, f, indent=4)
    
    # 세션 초기화 후 메인 페이지로 리다이렉트
    session.clear()
    return redirect(url_for('main'))

# 서버 실행
if __name__ == '__main__':
    app.run(debug=True)
