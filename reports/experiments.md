# Experiments (holdout month 33, validation months [29, 30, 31, 32, 33])

## RQ4: clip the training target or only the predictions
LightGBM with baseline params, seed 42. Predictions are clipped to [0, 20] in both cases; RMSE is against the clipped target.

RMSE per fold
| fold | clipped | raw |
|---|---|---|
| 29 | 0.7711 | 0.9209 |
| 30 | 0.7065 | 0.7780 |
| 31 | 0.7529 | 0.8701 |
| 32 | 0.8613 | 0.9190 |
| 33 | 0.7996 | 0.8611 |

Mean and std over folds
| train_target | rmse_mean | rmse_std |
|---|---|---|
| clipped | 0.7783 | 0.0574 |
| raw | 0.8698 | 0.0582 |

Holdout RMSE by slice
| slice | clipped | raw |
|---|---|---|
| A | 0.8290 | 0.8698 |
| B | 0.3060 | 0.3113 |
| C | 1.7866 | 2.0109 |
| all | 0.7996 | 0.8611 |

Paired cluster bootstrap over items (1000 resamples), delta RMSE (raw - clipped) on holdout: mean 0.0618, 95% CI [0.0287, 0.0960], share of resamples with delta > 0: 1.000

Summary: mean RMSE over folds is 0.7783 (clipped train target) vs 0.8698 (raw train target). On holdout raw - clipped is 0.0618; the 95% CI excludes zero.

## RQ5: seed sensitivity
Seeds: [0, 1, 2, 3, 4]. Variants: default = baseline params; subsample = baseline params + {'subsample': 0.8, 'subsample_freq': 1, 'colsample_bytree': 0.8}.

Default LightGBM params: RMSE differs across seeds (max spread over folds 7.49e-03).

Per fold, across seeds
| variant | fold | seed_mean | seed_std | seed_min | seed_max |
|---|---|---|---|---|---|
| default | 29 | 0.7693 | 0.0018 | 0.7669 | 0.7715 |
| default | 30 | 0.7039 | 0.0034 | 0.6995 | 0.7070 |
| default | 31 | 0.7519 | 0.0012 | 0.7505 | 0.7535 |
| default | 32 | 0.8598 | 0.0008 | 0.8586 | 0.8608 |
| default | 33 | 0.7978 | 0.0015 | 0.7962 | 0.7994 |
| subsample | 29 | 0.7722 | 0.0008 | 0.7716 | 0.7735 |
| subsample | 30 | 0.7033 | 0.0024 | 0.6996 | 0.7060 |
| subsample | 31 | 0.7518 | 0.0032 | 0.7477 | 0.7558 |
| subsample | 32 | 0.8584 | 0.0007 | 0.8572 | 0.8590 |
| subsample | 33 | 0.7983 | 0.0013 | 0.7966 | 0.8001 |

Seed std vs fold std vs gap to lag1 (fold_std: std over folds of seed-averaged RMSE; gap: mean over folds of lag1 RMSE - lgbm RMSE)
| variant | mean_seed_std | max_seed_std | fold_std | lgbm_minus_lag1_gap | max_spread_across_seeds |
|---|---|---|---|---|---|
| default | 0.0017 | 0.0034 | 0.0577 | 0.2128 | 0.0075 |
| subsample | 0.0017 | 0.0032 | 0.0573 | 0.2125 | 0.0081 |

Summary: mean seed std per fold is 0.0017 (default) and 0.0017 (subsample); std over folds is 0.0577 / 0.0573; mean gap to lag1 is 0.2128 / 0.2125.

Runtime: 379 s
