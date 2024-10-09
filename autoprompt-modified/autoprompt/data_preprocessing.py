import pandas as pd

def preprocess_data(input_file):
    # CSV 파일 읽기
    data = pd.read_csv(input_file)

    # 'Personality Combination'을 5개의 독립적인 성격 지표로 분리합니다.
    data[['Negative/Positive', 'Aggressive/Peaceful', 'Conservative/Open-minded', 'Introverted/Extroverted', 'Unstable/Confident']] = data['Personality Combination'].str.extract(r'\((\d), (\d), (\d), (\d), (\d)\)')

    # 각 성격 지표를 정수형으로 변환하고 0부터 시작하도록 조정합니다.
    data['Negative/Positive'] = data['Negative/Positive'].astype(int) - 1
    data['Aggressive/Peaceful'] = data['Aggressive/Peaceful'].astype(int) - 1
    data['Conservative/Open-minded'] = data['Conservative/Open-minded'].astype(int) - 1
    data['Introverted/Extroverted'] = data['Introverted/Extroverted'].astype(int) - 1
    data['Unstable/Confident'] = data['Unstable/Confident'].astype(int) - 1

    # BERT 모델의 입력과 출력 준비
    data['input_text'] = data['Prompt']
    # 각 성격 지표를 레이블 컬럼에 분리해서 추가
    data['label1'] = data['Negative/Positive']
    data['label2'] = data['Aggressive/Peaceful']
    data['label3'] = data['Conservative/Open-minded']
    data['label4'] = data['Introverted/Extroverted']
    data['label5'] = data['Unstable/Confident']

    # 데이터셋을 train과 dev로 나누기 (예: 80%/20% 비율로 나누기)
    train_data = data.sample(frac=0.8, random_state=42)
    dev_data = data.drop(train_data.index)

    # TSV 파일로 저장 (헤더를 포함하여 저장)
    train_data[['input_text', 'label1', 'label2', 'label3', 'label4', 'label5']].to_csv('train.tsv', sep='\t', index=False, header=True)
    dev_data[['input_text', 'label1', 'label2', 'label3', 'label4', 'label5']].to_csv('dev.tsv', sep='\t', index=False, header=True)

    # CSV 파일로도 저장
    train_data[['input_text', 'label1', 'label2', 'label3', 'label4', 'label5']].to_csv('train.csv', index=False, header=True)
    dev_data[['input_text', 'label1', 'label2', 'label3', 'label4', 'label5']].to_csv('dev.csv', index=False, header=True)

# 전처리 실행
preprocess_data('C:/Users/junhe/Desktop/캡디/Chatbot-with-Context-and-Personality/autoprompt-modified/prompts.csv')
