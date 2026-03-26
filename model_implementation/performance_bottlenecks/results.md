----------- Random Forest -----------------
DS5 — EV Energy Consumption | Random Forest
  CV R2 (5-fold)  : 0.9118 +/- 0.0036
  RMSE            : 0.6500
  R2              : 0.9129

DS1 — Vehicle Emissions | Random Forest
  CV Accuracy (5-fold) : 0.4195 +/- 0.0085
  Accuracy             : 0.4310
  Weighted F1          : 0.3417

DS4 — EV Charging Patterns | Random Forest
  CV R2 (5-fold)  : -0.0456 +/- 0.0287
  RMSE            : 10.8610
  R2              : -0.0553

DS6 — World GDP & CO2 Emissions | Random Forest
  CV R2 (3-fold)  : 0.9675 +/- 0.0196
  RMSE            : 5.6751e+08
  R2              : 0.9940

---------- Lasso ---------------------
  DS5 — EV Energy Consumption | Lasso
  Optimal alpha   : 0.002759
  Non-zero coefs  : 19 / 25
  RMSE            : 0.5108
  R2              : 0.9462

  DS1 — Vehicle Emissions | Lasso (L1 Logistic Regression)
  Optimal C per class : [0.1274275 0.1274275 0.1274275]
  Accuracy            : 0.4420
  Weighted F1         : 0.2710

DS4 — EV Charging Patterns | Lasso
  Optimal alpha   : 0.440624
  Non-zero coefs  : 2 / 36
  RMSE            : 10.6448
  R2              : -0.0137

DS6 — World GDP & CO2 Emissions | Lasso
  Optimal alpha   : 10.000000
  Non-zero coefs  : 14 / 14
  RMSE            : 1.0101e+08
  R2              : 0.9998

  --------- Gradient Boosting ---------
  DS5 — EV Energy Consumption | Gradient Boosting
  Best params : {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200}
  RMSE        : 0.5612
  R2          : 0.9350

DS1 — Vehicle Emissions | Gradient Boosting
  Best params  : {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}
  Accuracy     : 0.4420
  Weighted F1  : 0.2978

DS4 — EV Charging Patterns | Gradient Boosting
  Best params : {'learning_rate': 0.05, 'max_depth': 3, 'n_estimators': 100}
  RMSE        : 10.6428
  R2          : -0.0133

DS6 — World GDP & CO2 Emissions | Gradient Boosting
  Best params : {'learning_rate': 0.05, 'max_depth': 2, 'n_estimators': 100}
  RMSE        : 5.1995e+08
  R2          : 0.9949