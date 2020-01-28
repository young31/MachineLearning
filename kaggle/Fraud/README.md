# IEEE-fraud detection kaggle competition

## brief memo
 - used model: tree(LGBM, XGBOOST)
   - catboost is not used because it has somewhat different hyper parameter
   - NN models(GAN, DNN etc.) are not useful for this competition
 - feature engineering: almost depend on test-value
   - all variables except DT are masked so use only test-result(i.e. t-test, importance test)
 - public LB: 0.950x(stacking)
   - with blend get to 0.953x
   
co-worker's git [here](https://github.com/JeesooHaa)
   
