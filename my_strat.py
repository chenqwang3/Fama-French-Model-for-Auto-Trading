from datetime import datetime, timedelta, timezone

from apis.finance_data import get_symbol_list
from apis.finance_data import get_stock_info_list
from apis.finance_data import get_kline
from apis.finance_data import get_index_info
from apis.trade import make_order
from config import STRATEGY_NAME
from run_strategy import AccountContext
from utils import abspath
from utils.logger_tools import get_general_logger
from pprint import pprint
import numpy as np
import pandas as pd

logger = get_general_logger(STRATEGY_NAME, path=abspath("logs"))


def market_value_weighted(stocks, MV, BM):
    select = stocks[(stocks['NEGOTIABLEMV'] == MV) & (stocks['BM'] == BM)]  # 选出市值为MV，账目市值比为BM的所有股票数据
    market_value = select['NEGOTIABLEMV'].values  # 对应组的全部市值数据
    mv_total = np.sum(market_value)  # 市值求和
    mv_weighted = [mv / mv_total for mv in market_value]  # 市值加权的权重
    stock_return = select['return'].values
    # 返回市值加权的收益率的和
    return_total = []
    for i in range(len(mv_weighted)):
        return_total.append(mv_weighted[i] * stock_return[i])
    return_total = np.sum(return_total)
    return return_total


def main(context: AccountContext):
    # 每月第一个交易日的09:40 定时执行algo任务（仿真和实盘时不支持该频率）
    # schedule(schedule_func=algo, date_rule='1m', time_rule='09:40:00')
    # 数据滑窗
    # context.date = 20
    # 设置开仓的最大资金量
    context.ratio = 0.8
    # 账面市值比的大/中/小分类
    context.BM_BIG = 3.0
    context.BM_MID = 2.0
    context.BM_SMA = 1.0
    # 市值大/小分类
    context.MV_BIG = 2.0
    context.MV_SMA = 1.0

    # 获取上一个交易日的日期
    # last_day = get_previous_trading_date(exchange='SHSE', date=context.now)
    # 获取4580支A股代码 (未选行业)
    stockA = [item['代码'] for item in get_symbol_list()]
    # 获取当天有交易的股票
    # not_suspended = get_history_instruments(symbols=context.stock300, start_date=last_day, end_date=last_day)
    # not_suspended = [item['symbol'] for item in not_suspended if not item['is_suspended']]
    fin = pd.DataFrame([item for item in get_stock_info_list(symbols=stockA)])  # 获取P/B和市值数据
    # 计算账面市值比,为P/B的倒数
    fin['市净率'] = (fin['市净率'] ** -1)

    # 计算市值的50%的分位点,用于后面的分类
    size_gate = fin['流通市值'].quantile(0.50)
    # 计算账面市值比的30%和70%分位点,用于后面的分类
    bm_gate = [fin['市净率'].quantile(0.30), fin['市净率'].quantile(0.70)]
    fin.index = fin['代码']

    # 设置存放股票收益率的list
    x_return = []
    # kline_end = datetime.now(timezone(timedelta(hours=8)))
    # kline_start = kline_end - timedelta(days=6) + timedelta(seconds=1)
    # 对未停牌的股票进行处理
    for symbol in stockA:
        # 获取近5日K线数据
        # kline = get_kline(
        #     symbol,
        #     kline_start.strftime("%Y-%m-%d %H:%M:%S"),
        #     kline_end.strftime("%Y-%m-%d %H:%M:%S"),
        #     "1d",  # 天级
        # )
        # 计算收益率，存放到x_return里面
        # stock_return = kline[-1]['close'] / kline[0]['close'] - 1
        stock_return = fin['涨跌幅'][symbol]  # 目前平台K线数据不完整，先用涨跌幅代替收益率?
        pb = fin[fin['代码'] == symbol]['市净率'][symbol]
        market_value = fin[fin['代码'] == symbol]['流通市值'][symbol]

        # 获取[股票代码， 股票收益率, 账面市值比的分类, 市值的分类, 流通市值]
        # 其中账面市值比的分类为：大（3）、中（2）、小（1）
        # 流通市值的分类：大（2）、小（1）
        if pb < bm_gate[0]:
            if market_value < size_gate:
                label = [symbol, stock_return, context.BM_SMA, context.MV_SMA, market_value]
            else:
                label = [symbol, stock_return, context.BM_SMA, context.MV_BIG, market_value]
        elif pb < bm_gate[1]:
            if market_value < size_gate:
                label = [symbol, stock_return, context.BM_MID, context.MV_SMA, market_value]
            else:
                label = [symbol, stock_return, context.BM_MID, context.MV_BIG, market_value]
        elif market_value < size_gate:
            label = [symbol, stock_return, context.BM_BIG, context.MV_SMA, market_value]
        else:
            label = [symbol, stock_return, context.BM_BIG, context.MV_BIG, market_value]
        if len(x_return) == 0:
            x_return = label
        else:
            x_return = np.vstack([x_return, label])

    # 将股票代码、 股票收益率、 账面市值比的分类、 市值的分类、 流通市值存为数据表
    stocks = pd.DataFrame(data=x_return, columns=['symbol', 'return', 'BM', 'NEGOTIABLEMV', 'mv'])
    stocks.index = stocks.symbol
    columns = ['return', 'BM', 'NEGOTIABLEMV', 'mv']
    for column in columns:
        stocks[column] = stocks[column].astype(np.float64)

    # 计算SMB.HML和市场收益率（市值加权法）
    smb_s = (market_value_weighted(stocks, context.MV_SMA, context.BM_SMA) +
             market_value_weighted(stocks, context.MV_SMA, context.BM_MID) +
             market_value_weighted(stocks, context.MV_SMA, context.BM_BIG)) / 3
    # 获取大市值组合的市值加权组合收益率
    smb_b = (market_value_weighted(stocks, context.MV_BIG, context.BM_SMA) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_MID) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_BIG)) / 3
    smb = smb_s - smb_b
    # 获取大账面市值比组合的市值加权组合收益率
    hml_b = (market_value_weighted(stocks, context.MV_SMA, 3) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_BIG)) / 2
    # 获取小账面市值比组合的市值加权组合收益率
    hml_s = (market_value_weighted(stocks, context.MV_SMA, context.BM_SMA) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_SMA)) / 2
    hml = hml_b - hml_s

    # 获取市场收益率
    market_return = get_index_info('ID.000002')['涨跌幅']
    coff_pool = []
    # 对每只股票进行回归获取其alpha值
    for stock in stocks.index:
        x_value = np.array([[market_return], [smb], [hml], [1.0]])
        y_value = np.array([stocks['return'][stock]])
        # OLS估计系数
        coff = np.linalg.lstsq(x_value.T, y_value, rcond=None)[0][3]
        coff_pool.append(coff)
    # 获取alpha最小并且小于0的10只的股票进行操作(若少于10只则全部买入)
    stocks['alpha'] = coff_pool
    stocks = stocks[stocks.alpha < 0].sort_values(by='alpha').head(10)
    symbols_pool = stocks.index.tolist()
    logger.info(symbols_pool)
    # positions = context.positions["avaliable"]
    # 平不在标的池的股票
    # for position in positions:
    #     symbol = position['symbol']
    #     # if symbol not in symbols_pool:
    #     #     make_order(symbol, "market", 'sell', position)
    #     #     logger.info('市价单平不在标的池的', symbol)
    #     # 获取股票的权重
    #     percent = context.ratio / len(symbols_pool)
    #     # 买在标的池中的股票
    #     for symbol in symbols_pool:
    #         # make_order(symbol, "market", 'buy', position)
    #         logger.info(symbol, '以市价单调多仓到仓位', percent)
