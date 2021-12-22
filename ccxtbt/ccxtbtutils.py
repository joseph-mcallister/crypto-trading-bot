import json

import backtrader as bt

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


def get_config(params_path: str):
    with open(params_path, 'r') as f:
        params = json.load(f)
        return {'apiKey': params["binance"]["apikey"],
                'secret': params["binance"]["secret"],
                'enableRateLimit': True,
                }
