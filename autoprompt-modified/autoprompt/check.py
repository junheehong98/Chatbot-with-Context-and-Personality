import pandas as pd
import ast

# TSV 파일 읽기
train_tsv = pd.read_csv('train.tsv', sep='\t')
print(train_tsv.head())

# 문자열 형태의 labels를 실제 리스트로 변환하고, 각 요소가 0, 1, 2 내에 있는지 확인
def is_valid_label(label):
    try:
        # 문자열을 리스트로 변환
        label_list = ast.literal_eval(label)
        # 모든 요소가 0, 1, 2 내에 있는지 확인
        return all(int(i) in [0, 1, 2] for i in label_list)
    except:
        return False

# 유효하지 않은 레이블 찾기
invalid_labels = train_tsv[~train_tsv['labels'].apply(is_valid_label)]
print("Invalid labels in train.tsv:")
print(invalid_labels)
