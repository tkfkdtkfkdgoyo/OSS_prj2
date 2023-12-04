import pandas as pd

df = pd.read_csv('2019_kbo_for_kaggle_v2.csv', index_col=0)

#label에 관한 print the top 10 players
def top10PlayersByYear(dataset_df):
    years = [2015, 2016, 2017, 2018]
    labels = ['H', 'avg', 'HR', 'OBP']
    for year in years:
        filtered_df = dataset_df[dataset_df['year'] == year]
        for label in labels:
            top_10_players = filtered_df.sort_values(label, ascending = False).head(10)
            print(f"{year}년 {label} 기준 Top 10 선수:")
            
            for i, player_name in enumerate(top_10_players.index, 1): #1은 인덱스 시작 숫자
                print(f"{i}등 - {player_name}")
                print()
# 함수 호출
top10PlayersByYear(df)

#print the player with the highest war(승리 기여도) by position(cp) in 2018
def highestWarPlayersByCp(dataset_df):
    positionInfo = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']
    filtered_df = dataset_df[dataset_df['year'] == 2018]

    print("각 포지션별 가장 높은 'war' 값을 가진 선수 출력")
    for pos in positionInfo:
        player = filtered_df[filtered_df['cp'] == pos].sort_values(by = 'war', ascending = False).head(1)
        print(f"{pos} : {player.index[0]}")
        print()
        
# 함수 호출 
highestWarPlayersByCp(df)

#salary와 가장 높은 상관관계를 가진 지표 (특점, 안타, 홈런, 타점, 도루, 승리 기여도, 타율, 출루율, 장타율)
def highCorrWithSalary(dataset_df):
    columns = ['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']
    select_df = dataset_df[columns]
    
    #salary와 다른 지표 간의 상관관계 계산
    corr_salary = select_df.corr()['salary'].drop('salary')
    
    # 각 기준별 상관관계 출력
    print("각 기준별 salary와의 상관관계:")
    for index, value in corr_salary.items():
        print(f"{index}: {value}")
    
    #상관관계가 가장 높은 지표
    highest_corr = corr_salary.idxmax()
    
    return highest_corr, corr_salary[highest_corr]
    
# 함수 호출 
highestCorr, corrValue = highCorrWithSalary(df)
print(f"salary과 가장 높은 상관관계를 가지는 지표: {highestCorr}, 상관계수: {corrValue}")