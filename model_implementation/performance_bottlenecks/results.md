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




  Second try:
  -------- Lasso -----------
DS5 — EV Energy Consumption | Lasso
  Optimal alpha   : 0.002270
  Non-zero coefs  : 19 / 25
  Train RMSE     : 0.5138  R2: 0.9447
  Val   RMSE     : 0.5013  R2: 0.9463
  Test  RMSE     : 0.5114  R2: 0.9461

  ------ Gradient Boosting --------
DS5 — EV Energy Consumption | Gradient Boosting
  Best params : {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 200}
  Train RMSE     : 0.4078  R2: 0.9652
  Val   RMSE     : 0.5519  R2: 0.9349
  Test  RMSE     : 0.5665  R2: 0.9338

DS1 — Vehicle Emissions (High vs Not High) | Gradient Boosting
  Best params  : {'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200}
  Train Accuracy : 0.8038  Weighted F1: 0.8010
  Val   Accuracy : 0.5045  Weighted F1: 0.4932
  Test  Accuracy : 0.5320  Weighted F1: 0.5214
  ------ Random Forest ---------
DS5 — EV Energy Consumption | Random Forest
  CV R2 (5-fold)  : 0.9071 +/- 0.0047
  Train RMSE     : 0.2428  R2: 0.9877
  Val   RMSE     : 0.6372  R2: 0.9133
  Test  RMSE     : 0.6703  R2: 0.9074

DS1 — Vehicle Emissions (High vs Not High) | Random Forest
  CV Accuracy (5-fold) : 0.5410 +/- 0.0076
  Train Accuracy : 1.0000  Weighted F1: 1.0000
  Val   Accuracy : 0.5210  Weighted F1: 0.4640
  Test  Accuracy : 0.5420  Weighted F1: 0.4818