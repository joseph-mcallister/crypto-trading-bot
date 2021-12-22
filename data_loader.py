from datetime import datetime, timedelta
import backtrader as bt
from ccxtbt import CCXTStore, broker_mapping, get_config


class DataLoaderStrategy(bt.Strategy):
    def __init__(self):
        print("Initializing")

    def next(self):
        for data in self.datas:
            print('{} - {} | O: {} H: {} L: {} C: {} V:{}'.format(data.datetime.datetime(),
                                                                         data._name, data.open[0],
                                                                         data.high[0], data.low[0],
                                                                         data.close[0], data.volume[0],
                                                                         ))


cerebro = bt.Cerebro(quicknotify=True)

# Add the strategy
cerebro.addstrategy(DataLoaderStrategy)

config = get_config('./params.json')

store = CCXTStore(exchange='binanceus', currency='USDT', config=config, retries=5, debug=False)

broker = store.getbroker(broker_mapping=broker_mapping)
cerebro.setbroker(broker)

# Get our data
# Drop newest will prevent us from loading partial data from incomplete candles
hist_start_date = datetime.utcnow() - timedelta(weeks=52*5)
btc_data = store.getdata(dataname='BTC/USDT', name="BTCUSDT",
                         timeframe=bt.TimeFrame.Minutes, fromdate=hist_start_date,
                         compression=1, ohlcv_limit=10000)

# Add the feed
cerebro.adddata(btc_data)

# Run the strategy
cerebro.run()
