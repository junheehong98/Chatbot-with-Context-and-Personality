import pandas as pd

# 두 개의 CSV 파일 경로 설정
file1 = 'labeled_dialogues1.csv'
file2 = 'labeled_dialogues2.csv'

# CSV 파일을 읽기
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 두 데이터프레임을 하나로 합치기
combined_df = pd.concat([df1, df2], ignore_index=True)

# 합친 데이터프레임을 새 CSV 파일로 저장
combined_df.to_csv('labeled_dialogues.csv', index=False)

print("두 개의 CSV 파일이 'combined_labeled_dialogues.csv'로 성공적으로 합쳐졌습니다.")
