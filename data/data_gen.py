import csv
import json
from openai import OpenAI
import openai
import re
from openai import ChatCompletion


client = OpenAI(
    # This is the default and can be omitted
    api_key="",
)


print("OpenAI API 버전:", openai.__version__)

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None

# 레이블을 얻는 함수 정의
def generate_labels(dialogue):
    context = [
        {
            "role": "system",
            "content": "You are a personal trait label generation expert. Your task is to analyze the given dialogue and assign binary labels (0 or 1) for each of the following personality traits based on the content of the dialogue."
        },
        {
            "role": "user",
            "content": f"""
Please analyze the following dialogue and assign binary labels (0 or 1) for each of the following personality traits:

- Negative (0) or Positive (1)
- Aggressive (0) or Peaceful (1)
- Conservative (0) or Open-minded (1)
- Introverted (0) or Extroverted (1)
- Unstable (0) or Confident (1)

Example:
{{
  "text": "Good morning , sir . Is there a bank near here ?",
  "Negative_or_Positive": 1,
  "Aggressive_or_Peaceful": 1,
  "Conservative_or_Open-minded": 1,
  "Introverted_or_Extroverted": 1,
  "Unstable_or_Confident": 1
}}

Provide the result in the following JSON format without any additional text:

{{
  "Negative_or_Positive": 0 or 1,
  "Aggressive_or_Peaceful": 0 or 1,
  "Conservative_or_Open-minded": 0 or 1,
  "Introverted_or_Extroverted": 0 or 1,
  "Unstable_or_Confident": 0 or 1
}}

Dialogue:
\"\"\"
{dialogue}
\"\"\"
"""
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 모델 이름
        messages=context,
        temperature=0,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
    )
    # reply = response.choices[0].message['content'].strip()

    reply = response.choices[0].message.content.strip()
    json_text = extract_json(reply)

    # Parse JSON
    if json_text:
        try:
            labels = json.loads(json_text)
        except json.JSONDecodeError:
            labels = None
    else:
        labels = None

    return labels



# JSON 파일에서 대화 데이터 읽기
with open('dialogue_data.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    dialogues = data['dialogues']

# 시작 지점 설정 (예: start_index = 5000부터 시작)
start_index = int(input("처리를 시작할 인덱스 번호를 입력하세요 (0부터 시작): "))

# 결과를 CSV 파일로 저장 (이어 쓰기 모드)
with open('labeled_dialogues.csv', 'a', newline='', encoding='utf-8') as csvfile:
    fieldnames = [
        'text',
        'Label1',
        'Label2',
        'Label3',
        'Label4',
        'Label5'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # CSV 파일이 비어 있으면 헤더를 씁니다.
    if csvfile.tell() == 0:
        writer.writeheader()

    # 각 대화에 대해 레이블 생성
    for i in range(start_index, len(dialogues)):
        dialogue = dialogues[i]
        labels = generate_labels(dialogue)
        if labels:
            row = {
                'text': dialogue,
                'Label1': labels.get('Negative_or_Positive', ''),
                'Label2': labels.get('Aggressive_or_Peaceful', ''),
                'Label3': labels.get('Conservative_or_Open-minded', ''),
                'Label4': labels.get('Introverted_or_Extroverted', ''),
                'Label5': labels.get('Unstable_or_Confident', '')
            }
            writer.writerow(row)
            print(f"Processed dialogue {i+1}/{len(dialogues)}")
        else:
            print(f"레이블을 파싱할 수 없습니다 (대화 인덱스 {i})")
            # 레이블이 없을 경우 빈 값으로 채웁니다.
            row = {
                'text': dialogue,
                'Label1': '',
                'Label2': '',
                'Label3': '',
                'Label4': '',
                'Label5': ''
            }
            writer.writerow(row)

print("CSV 파일 'labeled_dialogues.csv'가 생성되었습니다.")


# 이전 버전이라고 한다

'''
# 레이블을 얻는 함수 정의


def generate_labels(dialogue):
    prompt =f"""
        Please analyze the following dialogue and assign binary labels (0 or 1) for each of the following personality traits:
        
        - Negative (0) or Positive (1)
        - Aggressive (0) or Peaceful (1)
        - Conservative (0) or Open-minded (1)
        - Introverted (0) or Extroverted (1)
        - Unstable (0) or Confident (1)
        
        Example:                   
                
        {{
        
          "text" : "Good morning , sir . Is there a bank near here ?",
          "Label1": 1,
          "Label2": 1,
          "Label3": 1,
          "Label4": 1,
          "Label5": 1
        }}
                  
        
        Provide the result in the following JSON format without any additional text:
        
        {{
          "Negative_or_Positive": 0 or 1,
          "Aggressive_or_Peaceful": 0 or 1,
          "Conservative_or_Open-minded": 0 or 1,
          "Introverted_or_Extroverted": 0 or 1,
          "Unstable_or_Confident": 0 or 1
        }}
        
        Dialogue:
        \"\"\"
        {dialogue}
        \"\"\"
        """


    response = openai.ChatCompletion.create(
        model="gpt-4",  # 또는 사용할 모델명
        # gpt-4o-mini

        messages=[
            {
                "role": "system",
                "content": "You are a personal trait label generation expert. Your task is to analyze the given dialogue and assign binary labels (0 or 1) for each of the following personality traits based on the content of the dialogue."

            },
            {
                "role": "user",
                "content": prompt



            }
        ],
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    reply = response.choices[0].message.content.strip()
    json_text = extract_json(reply)

    # Parse JSON
    if json_text:
        try:
            labels = json.loads(json_text)
        except json.JSONDecodeError:
            labels = None
    else:
        labels = None

    return labels




# JSON 파일에서 대화 데이터 읽기
with open('dialogue_data_test.json', 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)
    dialogues = data['dialogues']

# 시작 지점 설정 (예: start_index = 5000부터 시작)
start_index = int(input("처리를 시작할 인덱스 번호를 입력하세요 (0부터 시작): "))

# 결과를 CSV 파일로 저장 (이어 쓰기 모드)
with open('labeled_dialogues.csv', 'a', newline='', encoding='utf-8') as csvfile:
    fieldnames = [
        'text',
        'Label1',
        'Label2',
        'Label3',
        'Label4',
        'Label5'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # CSV 파일이 비어 있으면 헤더를 씁니다.
    if csvfile.tell() == 0:
        writer.writeheader()

    # 각 대화에 대해 레이블 생성
    for i in range(start_index, len(dialogues)):
        dialogue = dialogues[i]
        labels = generate_labels(dialogue)
        if labels:
            row = {'text': dialogue,
                   'Label1': labels.get('Negative_or_Positive', ''),
                   'Label2': labels.get('Aggressive_or_Peaceful', ''),
                   'Label3': labels.get('Conservative_or_Open-minded', ''),
                   'Label4': labels.get('Introverted_or_Extroverted', ''),
                   'Label5': labels.get('Unstable_or_Confident', '')
            }
            writer.writerow(row)
            print(f"Processed dialogue {i+1}/{len(dialogues)}")
        else:
            print(f"레이블을 파싱할 수 없습니다 (대화 인덱스 {i})")
            # 레이블이 없을 경우 빈 값으로 채우기
            row = {
                'text': dialogue,
                'Label1': '',
                'Label2': '',
                'Label3': '',
                'Label4': '',
                'Label5': ''
            }
            writer.writerow(row)

print("CSV 파일 'labeled_dialogues.csv'가 생성되었습니다.")

'''