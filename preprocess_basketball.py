import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

bd=pd.read_csv('cbb.csv')

'''
CONF: 학교가 참가하는 체육 대회(A10 = Atlantic 10, ACC = Atlantic Coast Conference, AE = America East, Amer = American, ASun = ASUN, B10 = Big Ten, B12 = Big 12, BE = Big East , BSky = Big Sky, BSth = Big South, BW = Big West, CAA = Colonial Athletic Association, CUSA = Conference USA, Horz = Horizon League, Ivy = Ivy League, MAAC = Metro Atlantic Athletic Conference, MAC = Mid-American Conference , MEAC = Mid-Eastern Athletic Conference, MVC = Missouri Valley Conference, MWC = Mountain West, NEC = Northeast Conference, OVC = Ohio Valley Conference, P12 = Pac-12, Pat = Patriot League, SB = Sun Belt, SC = Southern 컨퍼런스, SEC = 사우스 이스턴 컨퍼런스, Slnd = 사우스랜드 컨퍼런스, 합계 = 서밋 리그, SWAC = 사우스웨스턴 애슬레틱 컨퍼런스, WAC = 웨스턴 애슬레틱 컨퍼런스,WCC = 웨스트 코스트 컨퍼런스)

G: 플레이한 게임 수

W: 승리한 게임 수

ADJOE: 조정된 공격 효율성(팀이 평균 디비전 I 수비에 대해 가질 수 있는 공격 효율성(100 소유물당 득점)의 추정치)

ADJDE: 조정된 수비 효율성(평균 디비전 I 공격에 대해 팀이 가질 수 있는 수비 효율성(포제션 100개당 허용되는 점수)의 추정치)

BARTHAG: 파워 레이팅(평균 디비전 I 팀을 이길 확률)

EFG_O: 효과적인 필드 골 퍼센티지 샷

EFG_D: 유효 필드 골 허용 비율

TOR: 허용 회전율(회전율)

TORD: 커밋된 이직률(훔치는 비율)

ORB: 공격 리바운드율

DRB : 공격 리바운드율 허용

FTR : 자유투 비율(주어진 팀이 자유투를 던지는 빈도)

FTRD: 자유투율 허용

2P_O: 2점슛 퍼센티지

2P_D: 2점슛 허용 비율

3P_O: 3점슛 퍼센티지

3P_D: 3점슛 허용 비율

ADJ_T: 조정된 템포(팀이 평균 디비전 I 템포로 플레이하려는 팀을 상대로 가질 템포(40분당 점유)의 추정치)

WAB: 거품 위의 승리 - 본선 진출이 불투명함

POSTSEASON: 주어진 팀이 탈락하거나 시즌이 끝난 라운드(R68 = 퍼스트 포, R64 = 64강, R32 = 32강, S16 = 스위트 16, E8 = 엘리트 에이트, F4 = 파이널 포, 2ND = 러너 -up, Champion = 해당 연도의 NCAA March Madness Tournament 우승자)

SEED: NCAA March Madness Tournament의 시드

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
bd.drop(['TEAM', 'WAB', 'YEAR','W','G'], axis=1, inplace=True)