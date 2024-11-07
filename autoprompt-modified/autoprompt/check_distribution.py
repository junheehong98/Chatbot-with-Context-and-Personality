import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기
df = pd.read_csv('labeled_dialogues.csv')

# 각 성격 특성 컬럼을 하나의 조합으로 결합
df['personality_combination'] = df[['Label1', 'Label2', 'Label3', 'Label4', 'Label5']].astype(str).agg('-'.join, axis=1)

# 성격 조합별 빈도 계산
personality_distribution = df['personality_combination'].value_counts()

# 분포 출력
print("성격 조합별 데이터 분포:")
print(personality_distribution)

# 분포를 막대 그래프로 시각화
plt.figure(figsize=(12, 6))
personality_distribution.plot(kind='bar')
plt.title("Distribution of Personality Combinations")
plt.xlabel("Personality Combination (Label1-Label2-Label3-Label4-Label5)")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
