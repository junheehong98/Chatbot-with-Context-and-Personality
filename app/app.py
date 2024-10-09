from flask import Flask, render_template, request, redirect, url_for, session
import openai
from openai import OpenAI
from prompt import SYSTEM_PROMPT_TEMPLATE, SITUATION_LIST, PERSONALITY_LIST
from utils import get_personality_id, generate_system_prompt, trim_chat_history

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

# 대화 페이지
@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if 'trim_history' not in session:
        session['chat_history'] = []
        session['trim_history'] = []  # 대화 기록 초기화

    personality_id = get_personality_id(session['selected_characteristics'])
    system_prompt = generate_system_prompt(SITUATIONS.index(session['selected_situation']), personality_id)

    if request.method == 'POST':
        user_input = request.form.get('user_input')
        if user_input.strip():
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

            # 대화 기록 정리 
            trim_chat_history(session)

    return render_template('chat.html', chat_history=session['chat_history'])

# 서버 실행
if __name__ == '__main__':
    app.run(debug=True)
