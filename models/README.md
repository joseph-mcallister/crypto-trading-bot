# Model Tracker

### Hourly model leaderboard
Testing on XBTUSD_60 binary classification. Baseline accuracy: 50.4%
- SMAV0 + XGBoostV0 (12/22/21)
  - Train: 57.0%, Test: 54.0%
  - max_depth=3, gamma=500, eval_metric="logloss"
- SMAV0 + MLPV0 (12/22/21)
  - Train: 56.9%, Test 53.6%
  - solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1, max_iter=70
