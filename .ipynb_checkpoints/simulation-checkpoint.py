import csv
import os
import traceback

import cbpro
import Historic_Crypto as hc
import pandas as pd
import multiprocessing
from matplotlib import pyplot as plt
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)

delta1Min = pd.Timedelta('1min')
delta1D = pd.Timedelta('1Days')
delta5D = pd.Timedelta('5days')
delta30D = pd.Timedelta('30days')
delta60D = pd.Timedelta('60days')
delta90D = pd.Timedelta('90days')


def download_product(symb):
    today = pd.Timestamp.today()
    minus_30D = today - delta30D
    minus_60D = today - delta60D
    minus_90D = today - delta90D
    lock = multiprocessing.Lock()
    hist1 = hc.HistoricalData(symb, 60, minus_90D.strftime('%Y-%m-%d-%H-%M'), minus_60D.strftime('%Y-%m-%d-%H-%M')).retrieve_data()
    hist2 = hc.HistoricalData(symb, 60, minus_60D.strftime('%Y-%m-%d-%H-%M'), minus_30D.strftime('%Y-%m-%d-%H-%M')).retrieve_data()
    hist3 = hc.HistoricalData(symb, 60, minus_30D.strftime('%Y-%m-%d-%H-%M')).retrieve_data()
    path1 = symb + '1.csv'
    path2 = symb + '2.csv'
    path3 = symb + '3.csv'
    hist1.to_csv(path1)
    hist2.to_csv(path2)
    hist3.to_csv(path3)


def simulate_mvg(path, width):
    print(path)
    # read crypto 25-day data
    tbl = pd.read_csv(path)
    # create Datetime column
    tbl['time'] = pd.to_datetime(tbl['time'])
    tbl.set_index(tbl['time'], inplace=True)
    # set start interval
    start = tbl.iloc[0]['time']
    # set end to 5 days later
    end = start + delta5D
    sub = tbl.loc[start: end]
    length = len(sub)

    # calculate moving average
    mvg = sub['close'].rolling(width).mean()
    mvg = pd.DataFrame(mvg)
    # calculate upper and lower limit
    delta_price = 0

    buy_points = []
    sell_points = []

    # tbl['close'].plot()

    while len(sub) == length:
        # get last crypto price
        close = sub.iloc[len(sub) - 1]['close']
        # get latest EMVG
        prior_mvg = mvg.iloc[len(mvg) - 2]['close']
        latest_mvg = mvg.iloc[len(mvg) - 1]['close']
        delta_mvg = latest_mvg - prior_mvg
        # if latest crypto price is lower than the lower limit
        # advance the 5-day period until this is false
        if (latest_mvg >= close) or (delta_mvg < 0):
            while ((latest_mvg >= close) or (delta_mvg < 0)) and (len(sub) == length):
                sub = advance_subset(tbl, sub.iloc[0].name, sub.iloc[len(sub) - 1].name)
                mvg = sub['close'].rolling(width).mean()
                mvg = pd.DataFrame(mvg)

                close = sub.iloc[len(sub) - 1]['close']
                prior_mvg = mvg.iloc[len(mvg) - 2]['close']
                latest_mvg = mvg.iloc[len(mvg) - 1]['close']
                delta_mvg = latest_mvg - prior_mvg

        if len(sub) == length:
            buy_price = close
            buy_points.append(mvg.iloc[len(sub) - 1].name)
            original_buy_price = buy_price
            sell_price = latest_mvg
        else:
            sell_price = close

        # until the current price of the crypto goes
        # below the sell price, the data keeps advancing
        while (close > sell_price) and (len(sub) == length):
            sub = advance_subset(tbl, sub.iloc[0].name, sub.iloc[len(sub) - 1].name)
            mvg = sub['close'].rolling(width).mean()
            mvg = pd.DataFrame(mvg)

            close = sub.iloc[len(sub) - 1]['close']
            sell_price = mvg.iloc[len(mvg) - 1]['close']

        if len(buy_points) > len(sell_points):
            sell_points.append(mvg.iloc[len(mvg) - 1].name)
            delta_price += (sell_price - buy_price)

    try:
        # plt.vlines(x=buy_points, ymin=min(tbl['close']), ymax=max(tbl['close']), colors='red')
        # plt.vlines(x=sell_points, ymin=min(tbl['close']), ymax=max(tbl['close']), colors='green')
        # plt.show()
        delta_price = delta_price / original_buy_price
        return delta_price
    except UnboundLocalError:
        return 0


def simulate_emvg(path):
    print(path)
    # read crypto 25-day data
    tbl = pd.read_csv(path)
    # create Datetime column
    tbl['time'] = pd.to_datetime(tbl['time'])
    tbl.set_index(tbl['time'], inplace=True)
    # set start interval
    start = tbl.iloc[0]['time']
    # set end to 5 days later
    end = start + delta5D
    sub = tbl.loc[start: end]
    length = len(sub)

    # calculate moving average
    emvg = sub['close'].ewm(span=7).mean()
    emvg = pd.DataFrame(emvg)

    delta_price = 0
    buy_points = []
    sell_points = []

    # tbl['close'].plot()

    while len(sub) == length:
        # get last crypto price
        close = sub.iloc[len(sub) - 1]['close']
        # get latest EMVG
        prior_emvg = emvg.iloc[len(emvg) - 2]['close']
        latest_emvg = emvg.iloc[len(emvg) - 1]['close']
        delta_emvg = latest_emvg - prior_emvg
        # if latest crypto price is lower than the lower limit
        # advance the 5-day period until this is false
        if (latest_emvg >= close) or (delta_emvg < 0):
            while ((latest_emvg >= close) or (delta_emvg < 0)) and (len(sub) == length):
                sub = advance_subset(tbl, sub.iloc[0].name, sub.iloc[len(sub) - 1].name)
                emvg = sub['close'].ewm(span=7).mean()
                emvg = pd.DataFrame(emvg)

                close = sub.iloc[len(sub) - 1]['close']
                prior_emvg = emvg.iloc[len(emvg) - 2]['close']
                latest_emvg = emvg.iloc[len(emvg) - 1]['close']
                delta_emvg = latest_emvg - prior_emvg

        if len(sub) == length:
            buy_price = close
            buy_points.append(emvg.iloc[len(sub) - 1].name)
            original_buy_price = buy_price
            sell_price = latest_emvg

        # until the current price of the crypto goes
        # below the sell price, the data keeps advancing
        while (close > sell_price) and (len(sub) == length):
            sub = advance_subset(tbl, sub.iloc[0].name, sub.iloc[len(sub) - 1].name)
            emvg = sub['close'].ewm(span=7).mean()
            emvg = pd.DataFrame(emvg)

            close = sub.iloc[len(sub) - 1]['close']
            sell_price = emvg.iloc[len(emvg) - 1]['close']

        if len(buy_points) > len(sell_points):
            sell_points.append(emvg.iloc[len(emvg) - 1].name)
            delta_price += (sell_price - buy_price)

    try:
        # plt.vlines(x=buy_points, ymin=min(tbl['close']), ymax=max(tbl['close']), colors='red')
        # plt.vlines(x=sell_points, ymin=min(tbl['close']), ymax=max(tbl['close']), colors='green')
        # plt.show()
        returns = delta_price / original_buy_price
        avg_volume = tbl['volume'].mean()
        volume_psd = tbl['volume'].std() / avg_volume
        close_std = tbl['close'].std()
        avg_close = tbl['close'].mean()
        close_psd = close_std / avg_close
        return [returns, close_psd, avg_volume, volume_psd]
    except UnboundLocalError:
        return 0


def advance_subset(tbl, date1, date2):
    temp = tbl.reset_index(drop=True)
    start = temp.index[temp['time'] == date1].tolist()[0] + 1
    end = temp.index[temp['time'] == date2].tolist()[0] + 2
    sub = temp.iloc[start: end]
    sub.set_index(sub['time'], inplace=True)
    return sub


# this function calculates the sell price of a crypto
# a sell price an increase, however it cannot decrease
def calc_sell_price(close, mvg, old_sell_price=0):
    row = len(mvg) - 1
    if close > mvg.iloc[row]['upper']:
        sell_price = mvg.iloc[row]['upper']
    elif close > mvg.iloc[row]['close']:
        sell_price = mvg.iloc[row]['close']
    else:
        sell_price = mvg.iloc[row]['lower']
    if sell_price > old_sell_price:
        return sell_price
    else:
        return old_sell_price


def calc_psd(path):
    tbl = pd.read_csv(path)
    std = tbl['close'].std()
    avg = tbl['close'].mean()
    psd = std / avg
    return psd


def optimize_mvg(width):
    path = 'paths.txt'
    deltas = []
    names = []
    volatility = []
    with open(path, 'r') as f:
        paths = f.read().split('\n')
        for p in paths:
            if p != "":
                deltas.append(simulate_mvg(p, width))
                names.append(p.replace('.csv', ''))
                volatility.append(calc_psd(p))

    # write all data to csv
    deltas = pd.Series(deltas)
    names = pd.Series(names)
    volatility = pd.Series(volatility)
    frame = {'Crypto': names, 'Profit': deltas, 'Volatility': volatility}
    results = pd.DataFrame(frame)
    results.to_csv('results_mvg_' + str(width) + '.csv')


def optimize_emvg():
    path = 'paths2.txt'
    deltas = []
    names = []
    volatility = []
    volume = []
    volume_psd = []
    with open(path, 'r') as f:
        paths = f.read().split('\n')
        for p in paths:
            if p != "":
                for i in range(1,4):
                    results = simulate_emvg(p + str(i) + '.csv')
                    deltas.append(results[0])
                    names.append(p + ' ' +str(i))
                    volatility.append(results[1])
                    volume.append(results[2])
                    volume_psd.append(results[3])

    # write all data to csv
    deltas = pd.Series(deltas)
    names = pd.Series(names)
    volatility = pd.Series(volatility)
    volume = pd.Series(volume)
    volume_psd = pd.Series(volume_psd)
    frame = {'Crypto': names, 'Return': deltas, 'Volatility': volatility, 'Volume': volume, 'Volume PSD': volume_psd}
    results = pd.DataFrame(frame)
    results.to_csv('results4_emvg_.csv')


def summarize():
    summary_tbl = pd.read_csv('results3_emvg_7.csv', index_col=0)
    line = 0
    for c in summary_tbl['Crypto']:
        df = pd.read_csv(c + ".csv")
        summary_tbl['Volume'][line] = df['volume'].mean()
        summary_tbl['Volume Percent Deviation'][line] = df['volume'].std() / summary_tbl['Volume'][line]
        line += 1
    summary_tbl.to_csv('results3_emvg_7.csv')


def visualize(path):
    tbl = pd.read_csv(path)
    tbl['time'] = pd.to_datetime(tbl['time'])
    tbl.set_index(tbl['time'], inplace=True)
    tbl['close'].plot(label='Close')
    tbl['CMVG'] = tbl['close'].expanding().mean()
    tbl['CMVG'].plot(label='CMVG')
    plt.legend()
    plt.show()


def filter_files():
    files = os.listdir()
    for f in files:
        if f.endswith('.csv'):
            if not (f.endswith('-USD.csv')) and not ('results' in f):
                os.remove(f)
    files = os.listdir()
    for f in files:
        if f.endswith('-USD.csv'):
            with open('paths.txt', 'a') as p:
                p.write(f + '\n')

def test():
    today = pd.Timestamp.today()
    minus_5D = today - delta5D
    hist = hc.HistoricalData('BTC-USD', 60, minus_5D.strftime('%Y-%m-%d-%H-%M')).retrieve_data()
    hist.reset_index(inplace=True)
    target_time = hist.iloc[len(hist) - 1]['time'] + delta1Min
    while True:
        if pd.Timestamp.today() >= target_time:
            print(pc.get_product_ticker('BTC-USD'))
            target_time = target_time + delta1Min

if __name__ == '__main__':
    data = open('cbapi.txt', 'r').read().splitlines()

    public = data[0]
    passphrase = data[1]
    secret = data[2]

    pc = cbpro.PublicClient()
    ac = cbpro.AuthenticatedClient(public, passphrase, secret)

    #f = open('paths3.txt', 'r')
    #cryptos = f.read().split('\n')

    #pool = multiprocessing.Pool(8)
    #pool.map(download_product, cryptos)
    #optimize_emvg()

    test()
