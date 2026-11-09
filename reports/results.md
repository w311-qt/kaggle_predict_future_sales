# Baseline results (holdout month 33)

## RMSE per fold (diagnostic folds [22] are not in the mean)
| fold | zeros | lag1 | lgbm |
|---|---|---|---|
| 22 | 1.3618 | 1.2621 | 0.9605 |
| 29 | 1.0924 | 0.9719 | 0.7711 |
| 30 | 1.0243 | 0.8903 | 0.7065 |
| 31 | 1.1010 | 0.9441 | 0.7529 |
| 32 | 1.1840 | 1.0640 | 0.8613 |
| 33 | 1.1095 | 1.0762 | 0.7996 |

## Mean and std over validation months [29, 30, 31, 32, 33]
| model | rmse_mean | rmse_std | mae_mean | mae_std |
|---|---|---|---|---|
| zeros | 1.1022 | 0.0568 | 0.2666 | 0.0123 |
| lag1 | 0.9893 | 0.0795 | 0.3019 | 0.0110 |
| lgbm | 0.7783 | 0.0574 | 0.2983 | 0.0110 |

## Holdout by slice (A: pair seen, B: item seen elsewhere, C: new item)
| model | slice | rows_share | rmse | sq_err_share |
|---|---|---|---|---|
| zeros | A | 0.5241 | 1.2602 | 0.6761 |
| zeros | B | 0.4002 | 0.3352 | 0.0365 |
| zeros | C | 0.0757 | 2.1614 | 0.2874 |
| lag1 | A | 0.5241 | 1.2038 | 0.6557 |
| lag1 | B | 0.4002 | 0.3352 | 0.0388 |
| lag1 | C | 0.0757 | 2.1614 | 0.3055 |
| lgbm | A | 0.5241 | 0.8290 | 0.5633 |
| lgbm | B | 0.4002 | 0.3060 | 0.0586 |
| lgbm | C | 0.0757 | 1.7866 | 0.3781 |

## Holdout: logged pairs only vs full test-like grid
| model | rmse_full_grid | rmse_logged_only |
|---|---|---|
| zeros | 1.1095 | 2.9797 |
| lag1 | 1.0762 | 2.6073 |
| lgbm | 0.7996 | 1.9331 |

## Cluster bootstrap over items (1000 resamples), 95% CI of RMSE
| model | ci_low | ci_high |
|---|---|---|
| zeros | 0.9926 | 1.2281 |
| lag1 | 0.9656 | 1.1733 |
| lgbm | 0.7390 | 0.8620 |

Delta RMSE (lgbm - lag1): mean -0.2744, 95% CI [-0.3325, -0.2138], share of resamples with delta < 0: 1.000

## Largest lgbm errors on holdout per slice
| slice | shop_id | item_id | category | y | pred_lgbm | pred_lag1 |
|---|---|---|---|---|---|---|
| A | 39 | 21811 | 37 | 20.0000 | 0.0303 | 0.0000 |
| B | 39 | 16629 | 37 | 20.0000 | 0.0083 | 0.0000 |
| C | 42 | 13241 | 47 | 20.0000 | 0.6548 | 0.0000 |
