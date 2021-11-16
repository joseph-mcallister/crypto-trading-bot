from ccxtbt import CCXTStore
import backtrader as bt
from datetime import datetime, timedelta
import json


class TestStrategy(bt.Strategy):

    def __init__(self):

        self.sma = bt.indicators.SMA(self.data, period=21)
        self.btc_count = None
        self.usdt_count = None
        self.order_count = 0
        self.order = None

    def next(self):

        # Get cash and balance
        # New broker method that will let you get the cash and balance for
        # any wallet. It also means we can disable the getcash() and getvalue()
        # rest calls before and after next which slows things down.

        # NOTE: If you try to get the wallet balance from a wallet you have
        # never funded, a KeyError will be raised! Change LTC below as approriate

        self.btc_count = self.broker.get_wallet_balance('BTC')[1]
        self.usdt_count = self.broker.get_wallet_balance('USDT')[1]
        print('USDT balance: {}, BTC balance: {}'.format(self.usdt_count, self.btc_count))

        if self.live_data:
            cash, value = self.broker.get_wallet_balance('USDT')
            # For five minutes, alternate between buying BTC with 50% of available USDT, or selling all BTC available
            if not self.order and self.order_count < 5:
                if self.btc_count == 0:
                    size = self.usdt_count * 0.5 / self.datas[0].close
                    self.order = self.buy(size=size, exectype=bt.Order.Market)
                else:
                    self.order = self.sell(size=self.btc_count, exectype=bt.Order.Market)
        else:
            # Avoid checking the balance during a backfill. Otherwise, it will
            # Slow things down.
            cash = 'NA'

        for data in self.datas:
            print('{} - {} | Cash {} | O: {} H: {} L: {} C: {} V:{} SMA:{}'.format(data.datetime.datetime(),
                                                                                   data._name, cash, data.open[0],
                                                                                   data.high[0], data.low[0],
                                                                                   data.close[0], data.volume[0],
                                                                                   self.sma[0]))

    def notify_data(self, data, status, *args, **kwargs):
        dn = data._name
        dt = datetime.now()
        msg = 'Data Status: {}'.format(data._getstatusname(status))
        print(dt, dn, msg)
        if data._getstatusname(status) == 'LIVE':
            self.live_data = True
        else:
            self.live_data = False

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            self.order_count += 1
            if order.isbuy():
                print('BUY EXECUTED')
            elif order.issell():
                print('SELL EXECUTED')

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None


with open('./params.json', 'r') as f:
    params = json.load(f)

cerebro = bt.Cerebro(quicknotify=True)

# Add the strategy
cerebro.addstrategy(TestStrategy)

# Create our store
config = {'apiKey': params["binance"]["apikey"],
          'secret': params["binance"]["secret"],
          'enableRateLimit': True,
          }

# IMPORTANT NOTE - Kraken (and some other exchanges) will not return any values
# for get cash or value if You have never held any BNB coins in your account.
# So switch BNB to a coin you have funded previously if you get errors
store = CCXTStore(exchange='binance', currency='USDT', config=config, retries=5, debug=False, sandbox=True)

# Get the broker and pass any kwargs if needed.
# ----------------------------------------------
# Broker mappings have been added since some exchanges expect different values
# to the defaults. Case in point, Kraken vs Bitmex. NOTE: Broker mappings are not
# required if the broker uses the same values as the defaults in CCXTBroker.
broker_mapping = {
    'order_types': {
        bt.Order.Market: 'market',
        bt.Order.Limit: 'limit',
        bt.Order.Stop: 'stop-loss',  # stop-loss for kraken, stop for bitmex
        bt.Order.StopLimit: 'stop limit'
    },
    'mappings': {
        'closed_order': {
            'key': 'status',
            'value': 'closed'
        },
        'canceled_order': {
            'key': 'result',
            'value': 1}
    }
}

broker = store.getbroker(broker_mapping=broker_mapping)
cerebro.setbroker(broker)

# Get our data
# Drop newest will prevent us from loading partial data from incomplete candles
hist_start_date = datetime.utcnow() - timedelta(minutes=50)
btc_data = store.getdata(dataname='BTC/USDT', name="BTCUSDT",
                     timeframe=bt.TimeFrame.Minutes, fromdate=hist_start_date,
                     compression=1, ohlcv_limit=50, drop_newest=True)  # , historical=True)

# Add the feed
cerebro.adddata(btc_data)

# Run the strategy
cerebro.run()
