from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import openai
from openai import OpenAI
from autoprompt.create_trigger_user import run_model
from app.prompt import SYSTEM_PROMPT_TEMPLATE, SITUATION_LIST
from app.utils import get_personality_id, generate_system_prompt, trim_chat_history
import json
import os

app = Flask(__name__)
app.secret_key = 'supersecretkey'

# OpenAI API 키 설정
openai.api_key = ''
client = OpenAI(api_key="secret_key")

# 상황 리스트
SITUATIONS = ["Subway", "Lunch", "Library", "Cafe", "Class Break"]

# 성격 특성 목록 (각 특성마다 3가지 선택)
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
        # Retrieve the selected situation and characteristics from the form
        selected_situation = request.form.get('situation')

        selected_characteristics = []

        # Process each characteristic pair
        for i, (option0, option1) in enumerate(characteristics):
            selected = request.form.get(f'characteristic{i}')  # Get selected value
            if selected == option0:
                selected_characteristics.append(0)  # Map to 0
            elif selected == option1:
                selected_characteristics.append(1)  # Map to 1

        # Store selected values in the session
        session['selected_situation'] = selected_situation
        session['selected_characteristics'] = selected_characteristics
        
        return redirect(url_for('chat'))

    return render_template('main.html', situations=SITUATIONS, characteristics=characteristics)

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    session.permanent = True
    if 'trim_history' not in session:
        session['chat_history'] = []
        session['trim_history'] = []  # 대화 기록 초기화

    personality_id = session['selected_characteristics']
    system_prompt = generate_system_prompt(SITUATIONS.index(session['selected_situation']))

    if request.method == 'POST':
        user_input = request.json.get('user_input')

        # autoprompt 사용
        trigger_token = run_model(user_input, session['selected_characteristics'])
        user_input_tk = user_input + " " + " ".join(trigger_token["best_trigger_tokens"])

        # # autoprompt 미사용
        # user_input_tk = user_input 

        if user_input_tk and user_input_tk.strip():  
            conversation = [{"role": "system", "content": system_prompt}]
            for user_msg, bot_msg in session['trim_history']:
                conversation.append({"role": "user", "content": user_msg})
                conversation.append({"role": "assistant", "content": bot_msg})

            conversation.append({"role": "user", "content": user_input_tk})

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
            session['trim_history'].append((user_input_tk, bot_response))

            trim_chat_history(session)

            return jsonify({"bot_msg": bot_response})

    return render_template('chat.html', chat_history=session['chat_history'])

# 유저 평가를 저장할 딕셔너리
ratings_dict = {}

# 평가 처리
@app.route('/evaluate', methods=['POST'])
def evaluate():
    bot_msg = request.form.get('bot_msg')
    rating = request.form.get('rating')
    
    selected_situation = session.get('selected_situation')
    selected_characteristics = tuple(session.get('selected_characteristics')) 
    
    key = (selected_situation, selected_characteristics)
    
    key_str = str(key)
    
    if key_str not in ratings_dict:
        ratings_dict[key_str] = []
    
    ratings_dict[key_str].append([bot_msg, rating])
    
    return render_template('chat.html', chat_history=session['chat_history'])

# 유저가 대화 종료 시 데이터 저장
@app.route('/end_chat', methods=['POST'])
def end_chat():
    selected_characteristics = session.get('selected_characteristics')
    file_path = f'chat_ratings_{selected_characteristics}.json'
    
    with open('./app/ratings.json', 'w') as f:
        json.dump(ratings_dict, f, indent=4)  
    
    session.clear()
    return redirect(url_for('main'))

if __name__ == '__main__':
    app.run(debug=True)
