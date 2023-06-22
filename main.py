import csv
import hashlib
import hmac
import http
import math
import smtplib
import time
import traceback
from multiprocessing import Process
import numpy as np
import pandas as pd
import json
from threading import Thread
from websocket import create_connection, WebSocketConnectionClosedException
import requests


class Client:
    def __init__(self):
        data = open('cbapi.txt', 'r').read().splitlines()

        self.public = data[0]
        self.passphrase = data[1]
        self.secret = data[2]

    def sign_message(self, request) -> requests.Request:
        '''Signs the request'''

        api_key = self.public
        api_secret = self.secret

        timestamp = str(int(time.time()))
        body = (request.body or b"").decode()
        url = request.path_url.split("?")[0]
        message = f"{timestamp}{request.method}{url}{body}"
        signature = hmac.new(api_secret.encode("utf-8"), message.encode("utf-8"), digestmod=hashlib.sha256).hexdigest()

        request.headers.update(
            {
                "CB-ACCESS-SIGN": signature,
                "CB-ACCESS-TIMESTAMP": timestamp,
                "CB-ACCESS-KEY": api_key,
                "Content-Type": "application/json",
            }
        )

        return request

    def get_candles(self, product: str, start: int, end: int):
        payload = {}
        resp = requests.get(
            f"https://api.coinbase.com/api/v3/brokerage/products/{product}/candles?start={start}&end={end}&granularity=ONE_MINUTE",
            params=payload, auth=self.sign_message)
        return resp.json()

    def list_accounts(self):
        payload = {}
        resp = requests.get("https://api.coinbase.com/api/v3/brokerage/accounts", params=payload,
                            auth=self.sign_message)
        return resp.json()

    def list_products(self):
        payload = {}
        resp = requests.get("https://api.coinbase.com/api/v3/brokerage/products?product_type=SPOT", params=payload,
                            auth=self.sign_message)
        return resp.json()

    def post_sell_order(self, product: str, base_size: float):
        timestamp = str(int(time.time()))
        method = "POST"
        path = "/api/v3/brokerage/orders"
        payload = json.dumps({
            "client_order_id": str(np.random.randint(2 ** 31)),
            "product_id": product,
            "side": "SELL",
            "order_configuration": {
                "market_market_ioc": {
                    "base_size": str(base_size)
                }
            }
        })
        message = f"{timestamp}{method}{path}{payload}"
        signature = hmac.new(self.secret.encode('utf-8'), message.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()

        headers = {
            'CB-ACCESS-KEY': self.public,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-SIGN': signature,
            'accept': 'application/json'
        }

        conn = http.client.HTTPSConnection("api.coinbase.com")
        conn.request(method, path, payload, headers)
        res = conn.getresponse()
        data = res.read()
        return json.loads(data.decode("utf-8"))

    def post_buy_order(self, product: str, quote_size: float):
        timestamp = str(int(time.time()))
        method = "POST"
        path = "/api/v3/brokerage/orders"
        payload = json.dumps({
            "client_order_id": str(np.random.randint(2 ** 31)),
            "product_id": product,
            "side": "BUY",
            "order_configuration": {
                "market_market_ioc": {
                    "quote_size": str(quote_size)
                }
            }
        })
        message = f"{timestamp}{method}{path}{payload}"
        signature = hmac.new(self.secret.encode('utf-8'), message.encode('utf-8'), digestmod=hashlib.sha256).hexdigest()

        headers = {
            'CB-ACCESS-KEY': self.public,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-SIGN': signature,
            'accept': 'application/json'
        }

        conn = http.client.HTTPSConnection("api.coinbase.com")
        conn.request(method, path, payload, headers)
        res = conn.getresponse()
        data = res.read()
        return json.loads(data.decode("utf-8"))


class EmailAlert:
    def __init__(self):
        self.login = "kletis0419@gmail.com"
        self.password = "ywtlpsaouwyskiyy"
        self.receiver = 'mcmillar@purdue.edu'

    def send_alert(self, msg):
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(self.login, self.password)
            message = f"""\
Subject: An error has occured with Crypto Bot.
To: {self.receiver}
From: {self.login}

An error has occured:
{msg}"""

            server.sendmail(self.login, self.receiver, message)


class Product(Process):
    def __init__(self, product_id):
        Process.__init__(self)
        self._active = False
        self._lower_sell_limit = 0
        self._upper_sell_limit = 0
        self._client = Client()
        self._close = None
        self._crossed = False
        self._ema = None
        self.ema_to_close = None
        self._hist = None
        self._id = product_id
        self._quote_increment, self._base_increment = self.get_increments()
        self._base_currency, self._quote_currency = self._id.split('-')
        self._emailer = EmailAlert()

    def init_websocket(self):
        ws = None
        thread = None
        thread_running = False
        thread_keepalive = None

        def websocket_thread():
            nonlocal ws

            api_key = self._client.public
            api_secret = self._client.secret

            channel = "ticker"
            timestamp = str(int(time.time()))
            message = f"{timestamp}{channel}{self._id}"
            signature = hmac.new(api_secret.encode("utf-8"), message.encode("utf-8"),
                                 digestmod=hashlib.sha256).hexdigest()

            ws = create_connection("wss://advanced-trade-ws.coinbase.com")
            ws.send(
                json.dumps(
                    {
                        "type": "subscribe",
                        "product_ids": [
                            "BTC-USD",
                        ],
                        "channel": channel,
                        "api_key": api_key,
                        "timestamp": timestamp,
                        "signature": signature,
                    }
                )
            )

            thread_keepalive.start()
            while not thread_running:
                try:
                    ticker_data = ws.recv()
                    if ticker_data != "":
                        msg = json.loads(ticker_data)
                    else:
                        msg = {}
                except ValueError:
                    tb = traceback.format_exc()
                    self._emailer.send_alert(tb)
                    self.init_websocket()
                except Exception:
                    tb = traceback.format_exc()
                    self._emailer.send_alert(tb)
                    self.init_websocket()
                else:
                    if "result" not in msg:
                        try:
                            timestamp = pd.Timestamp(msg['timestamp'])
                            price = float(msg['events'][0]['tickers'][0]['price'])
                            self._hist.loc[timestamp] = [price, 0]
                            if self._hist.index[-1] - self._hist.index[0] > pd.Timedelta('5days'):
                                with open(f'{self._id}_ticker_history.csv', mode='a', encoding='utf-8', newline='') as f:
                                    writer = csv.writer(f)
                                    row = self._hist.loc[self._hist.index[0]]
                                    writer.writerow([row.name, row['Close'], row['EMA']])
                                self._hist.drop(self._hist.index[0], inplace=True)
                            self._hist['EMA'] = self._hist['Close'].ewm(halflife='10 min', times=self._hist.index).mean()
                            self._ema = self._hist['EMA'][self._hist.index[-1]]
                            self._close = self._hist['Close'][self._hist.index[-1]]

                            print(f'{self._hist.index[-1]}: {self._close} {self._ema}')

                            if (self._ema > self._close) and (self._close > self._upper_sell_limit) and self._active:
                                print('SELL')
                                self.sell()
                            elif (self._close < self._ema) and (self._close < self._lower_sell_limit) and self._active:
                                print('SELL')
                                self.sell()
                            elif (self._ema < self._close) and not self._active:
                                print('BUY')
                                self.buy()
                        except KeyError:
                            pass

            try:
                if ws:
                    ws._close()
            except WebSocketConnectionClosedException:
                pass
            finally:
                thread_keepalive.join()

        def websocket_keepalive(interval=30):
            nonlocal ws
            while ws.connected:
                ws.ping("keepalive")
                time.sleep(interval)

        thread = Thread(target=websocket_thread)
        thread_keepalive = Thread(target=websocket_keepalive)
        thread.start()

    def run(self):
        try:
            end = pd.Timestamp.today(tz='UTC') + pd.Timedelta('1min')
            start = end - pd.Timedelta('5days')
            intervals = [start]

            while start != end:
                start = intervals[len(intervals) - 1] + pd.Timedelta('4hours')
                intervals.append(start)

            intervals.append(end)
            intervals = [int(t.timestamp()) for t in intervals]
            hists = []

            for i in range(len(intervals) - 2):
                hist = pd.DataFrame(self._client.get_candles(self._id, intervals[i], intervals[i + 1])['candles'])
                hist.columns = ["Date", "Low", "High", "Open", "Close", "Volume"]
                hist['Date'] = pd.to_datetime(hist['Date'], unit='s', utc=True)
                hist['Close'] = pd.to_numeric(hist['Close'])
                hist.sort_values(by='Date', ascending=True, inplace=True)
                hist.set_index('Date', inplace=True)
                hists.append(hist)

            self._hist = pd.concat(hists)[['Close']]
            self._hist['EMA'] = self._hist['Close'].ewm(halflife='7 min', times=self._hist.index).mean()

        except:
            tb = traceback.format_exc()
            self._emailer.send_alert(tb)

        self.init_websocket()

    def get_available_currency(self, currency: str) -> float:
        for account in self._client.list_accounts()['accounts']:
            if account['currency'] == currency:
                return float(account['available_balance']['value'])

    def get_increments(self):
        for product in self._client.list_products()['products']:
            if product['product_id'] == self._id:
                return len(product['quote_increment']) - 2, len(product['base_increment']) - 2

    def sell(self):
        base_size = self.get_available_currency(self._base_currency)
        base_size = math.floor(base_size * 10 ** self._base_increment) / 10 ** self._base_increment
        resp = self._client.post_sell_order(self._id, base_size)
        if not resp['success']:
            self._emailer.send_alert(resp['error_response'])
        else:
            with open(f'{self._id}_trade_history.csv', mode='a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                row = [self._hist.index[-1], self._hist.iloc[-1]['Close'], 'SELL', base_size]
                writer.writerow(row)
            self._active = False

    def buy(self):
        quote_size = self.get_available_currency(self._quote_currency)
        quote_size = math.floor(quote_size * 10 ** self._quote_increment) / 10 ** self._quote_increment
        resp = self._client.post_buy_order(self._id, quote_size)
        if not resp['success']:
            self._emailer.send_alert(resp['error_response'])
        else:
            with open(f'{self._id}_trade_history.csv', mode='a', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                row = [self._hist.index[-1], self._hist.iloc[-1]['Close'], 'BUY', quote_size]
                writer.writerow(row)
            self._active = True
            self._lower_sell_limit = .956 * self._close
            self._upper_sell_limit = 1.012 * self._close


if __name__ == '__main__':
    btc = Product('BTC-USD')
    btc.start()
