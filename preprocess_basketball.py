import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

bd=pd.read_csv('cbb.csv')

'''
TEAM : 참여하는 학교의 이름
CONF : 소속 지역
G : 게임수
W : 승리한 게임수
ADJOE : 조정된 공격 효율성(평균 디비전 I 방어에 대해 팀이 가질 공격 효율성(점유율당 득점)의 추정치)
ADJDE : 수정된 방어 효율성(평균 디비전 I 공격에 대해 팀이 가질 방어 효율성(점유율당 실점)의 추정치)
BARTHAG : 전력 등급(평균 디비전 I 팀을 이길 가능성)
EFG_O : 유효슛 비율
EFG_D : 유효슛 허용 비율
TOR : 턴오버 비율(흐름 끊은 비율)
TORD : 턴오버 허용 비율(흐름 끊긴 비율)
ORB : 리바운드 차지 횟수
DRB : 리바운드 허용 횟수
FTR : 자유투 비율
FTRD : 자유투 허용 비율
2P_O : 2점 슛 성공 비율
2P_D : 2점 슛 허용 비율
3P_O : 3점 슛 성공 비율
3P_D : 3점 슛 허용 비율
ADJ_T : 조정된 템포(팀이 평균 디비전 I 템포로 플레이하려는 팀을 상대로 가질 템포(40분당 점유)의 추정치)
WAB : "Wins Above Bubble"은 NCAA 농구 대회의 예선 라운드에 참가하는 팀을 결정하는 데 사용되는 "버블"(일정 선) 기준에서 얼마나 높은 승리를 거두었는지를 나타내는 지표입니다.
POSTSEASON : 팀이 시즌을 마무리한 등수
SEED : NCAA 토너먼트에 참가하는 시드(등수)
YEAR : 시즌

'''

bd["Winning_rate"] = bd['W']/bd['G']
ps={
    "R68":68,
    "R64":64,
    "R32":32,
    "S16":16,
    "E8":8,
    "F4":4,
    "2ND":2,
    "Champion":1
}
bd['POSTSEASON'] = bd['POSTSEASON'].map(ps)
bd.fillna({'POSTSEASON':'Missed Tournament'}, inplace = True)
bd.fillna({'SEED':'Missed Tournament'}, inplace = True)
bd=pd.get_dummies(bd, columns=['CONF','SEED','POSTSEASON'])
bd.drop(['TEAM', 'YEAR','W','G'], axis=1, inplace=True)

# min-max 스케일링
from sklearn.preprocessing import MinMaxScaler
mm_scaler = MinMaxScaler()
X_train_scaled = mm_scaler.fit_transform(X_train)
X_test_scaled = mm_scaler.transform(X_test)