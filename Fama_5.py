import pandas as pd
import numpy as np
import datetime
import dateutil
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

#数据集：
# bm.csv 账面市值比（PB倒数）
# cap.csv 流通市值
# EBIT.csv 息税前利润
# Equity.csv 股东权益合计（不含少数股东）
# TotAsset.csv 总资产
# close.csv 收盘价

bm=pd.read_csv("bm.csv")
cap=pd.read_csv("cap.csv")
ebit=pd.read_csv("EBIT.csv")
equity=pd.read_csv("Equity.csv")
totasset=pd.read_csv("TotAsset.csv")
close=pd.read_csv("close.csv")

#设置索引
cap=cap.set_index("Date")
close=close.set_index("Date")
bm=bm.set_index("Date")
ebit=ebit.set_index("Date")
equity=equity.set_index("Date")
totasset=totasset.set_index("Date")

#提取构造时间序列，时间序列可以用来提取特定日期的数据
floatlist=cap.index
#cap的index为浮点型，先将其转化为整数再转化为str
datelist=pd.to_datetime(floatlist.astype("int").astype("str"))

#构造ALLapool可交易股票池
allapool=pd.DataFrame(index=floatlist,columns=cap.columns)
allapool[close>0]=1
allapool=allapool.fillna(0)
#allapool为构建的有效股票池（wind全A中可以交易的股票），除去了nan值，原因是这些值的股票可能未上市或者刚上市120天以内，因为上市带来新股溢价并且没有正式的财报数据，收益不稳定，所以这些股票应该设置为不可交易）。


#bm的index不为floatlist
bm.set_index(floatlist,inplace=True)
totasset.set_index(floatlist,inplace=True)

pro=(ebit/equity)*allapool #构造盈利能力指标、将盈利能力中不可交易的股票去除
cap=cap*allapool#将市值中不可交易的股票去除
bm=bm*allapool#将bm中不可交易的股票去除
totasset=totasset*allapool


#提取不同数据集相应月份的数据，对应五因子构造
cap_d=cap.set_index(datelist)
cap_m=cap_d.groupby([cap_d.index.year,cap_d.index.month]).tail(1)
cap_may=cap_m[cap_m.index.month==5]

bm_d=bm.set_index(datelist)
bm_m=bm_d.groupby([bm_d.index.year,bm_d.index.month]).tail(1)
bm_may=bm_m[bm_m.index.month==5]

pro_d=pro.set_index(datelist)
pro_m=pro_d.groupby([pro_d.index.year,pro_d.index.month]).tail(1)
pro_may=pro_m[pro_m.index.month==5]

totasset_d=totasset.set_index(datelist)
totasset_m= totasset_d.groupby([totasset_d.index.year,totasset_d.index.month]).tail(1)
totasset_dec =totasset_m[totasset_m.index.month==12]
inv = (totasset_dec.shift(-1)-totasset_dec)/totasset_dec
inv = pd.DataFrame(inv,index=totasset_d.index,columns=totasset.columns)
inv = inv.fillna(method='ffill') #method=ffill 为向下填充，在量化分析中会经常用到，另外一个常见的bfill为向上填充
inv_may = inv.groupby([inv.index.year,inv.index.month]).tail(1)
inv_may = inv_may[inv_may.index.month==5]


H=bm_may.apply(lambda x:x>= x.quantile(0.7),axis=1)#选取bm前百分之30的股票组合
M=bm_may.apply(lambda x:(x>=x.quantile(0.3))&(x<x.quantile(0.7)),axis=1)#选取bm值30%-70%的股票组合
L=bm_may.apply(lambda x:x<x.quantile(0.3),axis=1)#选取bm后百分之30的股票组合
#以下同上
B=cap_may.apply(lambda x:x>=x.quantile(0.5),axis=1)
S=cap_may.apply(lambda x:x<x.quantile(0.5),axis=1)

R=pro_may.apply(lambda x:x>=x.quantile(0.7),axis=1)
W=pro_may.apply(lambda x:x<x.quantile(0.3),axis=1)

A=inv_may.apply(lambda x:x>=x.quantile(0.7),axis=1)
C=inv_may.apply(lambda x:x<x.quantile(0.3),axis=1)



#构建不同的股票组合
BH=B&H
BM=B&M
BL=B&L
SH=S&H
SM=S&M
SL=S&L
SR=S&R
BR=R&B
SW=R&S
BW=B&W
SC=S&C
BC=B&C
SA=S&A
BA=B&A

#值为1的即持仓组合
BH=BH*1
BM=BM*1
BL=BL*1
SH=SH*1
SM=SM*1
SL=SL*1
SR=SR*1
SW=SW*1
BR=BR*1
BW=BW*1
SC=SC*1
BC=BC*1
SA=SA*1
BA=BA*1

#市值加权持仓函数,根据每年五月末的数据计算日频持仓
def get_score(stocklist):
    pos=pd.DataFrame(stocklist,index=cap_d.index,columns=cap.columns)
    pos=pos.fillna(method="ffill")
    pos=pos.set_index(floatlist)
    pos=pos*allapool
    score=((pos*cap).T/(pos*cap).sum(axis=1)).T#该步为根据市值加权计算持股比例
    return score

#各组合的持仓，因组合种类较多，该方法步骤较多可以优化
score_BH=get_score(BH)
score_BH=score_BH.loc[~score_BH.isna().all(axis=1)]

score_BM=get_score(BM)
score_BM=score_BM.loc[~score_BM.isna().all(axis=1)]

score_BL=get_score(BL)
score_BL=score_BL.loc[~score_BL.isna().all(axis=1)]

score_SH=get_score(SH)
score_SH=score_SH.loc[~score_SH.isna().all(axis=1)]

score_SM=get_score(SM)
score_SM=score_SM.loc[~score_SM.isna().all(axis=1)]

score_SL=get_score(SL)
score_SL=score_SL.loc[~score_SL.isna().all(axis=1)]

score_SR=get_score(SR)
score_SR=score_SR.loc[~score_SR.isna().all(axis=1)]

score_SW=get_score(SW)
score_SW=score_SW.loc[~score_SW.isna().all(axis=1)]

score_BR=get_score(BR)
score_BR=score_BR.loc[~score_BR.isna().all(axis=1)]

score_BW=get_score(BW)
score_BW=score_BW.loc[~score_BW.isna().all(axis=1)]

score_SC=get_score(SC)
score_SC=score_SC.loc[~score_SC.isna().all(axis=1)]

score_BC=get_score(BC)
score_BC=score_BC.loc[~score_BC.isna().all(axis=1)]

score_SA=get_score(SA)
score_SA=score_SA.loc[~score_SA.isna().all(axis=1)]

score_BA=get_score(BA)
score_BA=score_BA.loc[~score_BA.isna().all(axis=1)]

#累计收益 因组合较多该方法较为繁琐，可以适当优化，但思路不难
ret_SL=(close.loc[score_SL.index].pct_change().shift(-1)*score_SL).sum(axis=1)
ret_SL=(ret_SL+1).cumprod()#计算绝对收益率

ret_BL=(close.loc[score_BL.index].pct_change().shift(-1)*score_BL).sum(axis=1)
ret_BL=(ret_BL+1).cumprod()

ret_SM=(close.loc[score_SM.index].pct_change().shift(-1)*score_SM).sum(axis=1)
ret_SM=(ret_SM+1).cumprod()

ret_BM=(close.loc[score_BM.index].pct_change().shift(-1)*score_BM).sum(axis=1)
ret_BM=(ret_BM+1).cumprod()

ret_BH=(close.loc[score_BH.index].pct_change().shift(-1)*score_BH).sum(axis=1)
ret_BH=(ret_BH+1).cumprod()

ret_SH=(close.loc[score_SH.index].pct_change().shift(-1)*score_SH).sum(axis=1)
ret_SH=(ret_SH+1).cumprod()

ret_SR=(close.loc[score_SR.index].pct_change().shift(-1)*score_SR).sum(axis=1)
ret_SR=(ret_SR+1).cumprod()

ret_SW=(close.loc[score_SW.index].pct_change().shift(-1)*score_SW).sum(axis=1)
ret_SW=(ret_SW+1).cumprod()

ret_BR=(close.loc[score_BR.index].pct_change().shift(-1)*score_BR).sum(axis=1)
ret_BR=(ret_BR+1).cumprod()

ret_BW=(close.loc[score_BW.index].pct_change().shift(-1)*score_BW).sum(axis=1)
ret_BW=(ret_BW+1).cumprod()

ret_SC=(close.loc[score_SC.index].pct_change().shift(-1)*score_SC).sum(axis=1)
ret_SC=(ret_SC+1).cumprod()

ret_BC=(close.loc[score_BC.index].pct_change().shift(-1)*score_BC).sum(axis=1)
ret_BC=(ret_BC+1).cumprod()

ret_SA=(close.loc[score_SA.index].pct_change().shift(-1)*score_SA).sum(axis=1)
ret_SA=(ret_SA+1).cumprod()

ret_BA=(close.loc[score_BA.index].pct_change().shift(-1)*score_BA).sum(axis=1)
ret_BA=(ret_BA+1).cumprod()

#计算因子
SMB=(ret_SH+ret_SM+ret_SL)/3-(ret_BH+ret_BM+ret_BL)/3
HML=(ret_SH+ret_BH)/2-(ret_SL+ret_BL)/2
RMW=(ret_SR+ret_BR)/2-(ret_SW+ret_BW)/2
CMA=(ret_SC+ret_BC)/2-(ret_SA+ret_BA)/2


#导入已发布的因子检测
fama = pd.read_csv('fivefactor_daily.csv')
fama.index=fama["trddy"].str.replace("-","").astype("float")


#计算因子
fama_smb = fama.loc[SMB.index]
fama_smb = (fama['smb']+1).cumprod()

fama_hml = fama.loc[HML.index]
fama_hml = (fama['hml']+1).cumprod()

fama_rmw = fama.loc[RMW.index]
fama_rmw = (fama['rmw']+1).cumprod()

fama_cma = fama.loc[CMA.index]
fama_cma = (fama['cma']+1).cumprod()



c_smb=fama_smb.corr(SMB)#计算相关性
print(c_smb)
c_hml=fama_hml.corr(HML)
print(c_hml)
c_rmw=fama_rmw.corr(RMW)
print(c_rmw)
c_cma=fama_cma.corr(CMA)
print(c_cma)

#导入沪深300作为benchmark
csi_300 = pd.read_csv('IndexPrice500.csv',index_col=0)

def csi(df):
    df.index=datelist
    df=df["Close"].pct_change()
    df=(df+1).cumprod()
    return df

csi_300 = csi(csi_300)



#画图
def plot(a,b,c,d,na,nb,nc,nd):
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.xlabel("time")
    plt.ylabel("return")
    # ticks=[datetime(2017,5,31),datetime(2019,5,31),datetime(2020,5,31)]#
    ax=plt.subplot(221)
    plt.plot(csi_300,label="bmk500")
    a.index=pd.to_datetime(a.index.astype("int").astype("str"))
    plt.plot(a,label=na)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(365))
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend(loc=0)

    ax2=plt.subplot(222)
    plt.plot(csi_300, label="bmk500")
    b.index = pd.to_datetime(b.index.astype("int").astype("str"))
    plt.plot(b, label=nb)
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(365))
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend(loc=0)
    ax3=plt.subplot(223)
    plt.plot(csi_300, label="bmk500")
    c.index = pd.to_datetime(c.index.astype("int").astype("str"))
    plt.plot(c, label=nc)
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(365))
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend(loc=0)

    ax4=plt.subplot(224)
    plt.plot(csi_300, label="bmk500")
    d.index = pd.to_datetime(d.index.astype("int").astype("str"))
    plt.plot(d, label=nd)
    ax4.xaxis.set_major_locator(ticker.MultipleLocator(365))
    plt.xticks(rotation=45)
    plt.grid()
    plt.legend(loc=0)
    plt.show()


plot(ret_SL,ret_SH,ret_SA,ret_BH,"SL","SH","SA","BH")
#可以自己选择随意的组合回测，这里只回测了SL SH SA BH
