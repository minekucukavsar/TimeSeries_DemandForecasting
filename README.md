# TimeSeriesAnalysis_DemandForecasting

**Methods**

- Smoothing Methods

a/Single Exponential Smoothing(alfa)

b/Double Exponential Smoothing(alfa,beta)

c/Triple Exponential Smoothing/Holt-Winters(alfa,beta,gamma)


- Statistical Methods

a/AR(p),MA(q),ARMA(p,q)

b/ARIMA(p,d,q)

c/SARIMA(p,d,q)


**About Case**

*This case is the Store Item Demand Forecasting*

**Business Problem**

*A chain market wants a 3-month demand forecast for its 10 different stores and 50 different products.*

**About Data**

*Chain market's 5-year data includes information on 10 different stores and 50 different products.*



**LightGBM**

- LightGBM is a type of GBM developed to improve XGBoost's training time performance.

- When the number of hyperparameter combinations to be searched increases, the training time of XGBoost can be very long. LightGBM has been suggested as a solution.LightGBM performs in a shorter time.

- It has a Leaf-Wise growth strategy.

 **The most important lightgbm parameters**

 *num_leaves: maximum number of leaves on a tree*
 
 *learning_rate: shrinkage_rate, eta*
 
 *feature_fraction: It is the random subspace feature of RF. Number of variables to be considered randomly in each iteration (RF: random forest).*
 
 *max_depth: This parameter control max depth of each trained tree*
 
 *num_boost_round: n_estimators, number of boosting iterations. It should be around 10000-15000 at least.*

 *early_stopping_rounds: If the metric in the validation set does not progress at a certain early_stopping_rounds, that is, if the error does not decrease,     this parameter will stop training. It both shortens the train time and prevents overfitting.*
 
 *nthread: num_thread, nthread, nthreads, n_jobs*
