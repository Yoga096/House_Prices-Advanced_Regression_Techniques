# House Prices - Advanced Regression Techniques
Predict sales prices and practice feature engineering, RFs, and gradient boosting.  
kaggle: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
預測房價

## 成績

 * Test RMSE : 0.0.12656
 * Top 20% (814/4,496)


## 作法

1. ### EDA
 * 價格分布與常態分佈
 * 房價與其他變數的相關性
 * 欄位缺值比例
 * 數值類變數的分布情形


2. ### 預處理
 * 極端值訓練資料排除
 * 處理遺漏值
  * 缺值比例過高欄位 => 移除
  * 類別欄位: 該欄位訓練資料缺值3筆以上 => 補'none'，否則補眾數
  * 數值欄位: 訓練資料眾數為0補0、否則補中位數
  * 車庫年份 => 補建造年份
 * 製造特徵: 
    * na 欄位數量
    * 屋齡、整修、車庫年齡
    * 出售年份 => 各年齡層組別
 * One-Hot Encoding 
 

3. ### 標準化、切割訓練/測試/驗證集
 * 以訓練資料為基準將 Id, SalePrice 以外其他欄位標準化
 * 切割: train set => 85%, valid set => 15%
 * y_train、y_valid 取log 方便之後計算


4. ### 單一模型訓練、預測 (Grid search 找最佳參數)
 * OLS
 * Ridge
 * Lasso
 * XGBoost
 * CatBoost

5. ### 集成模型
 * Voting: 除OLS以外的4個模型平均or 加權平均(VotingRegressor)
 * Stacking: 除OLS以外的4個模型 + LinearRegression (StackingRegressor)

6. ### Neural Network Model (PyTorch)
 * Model:　Linear + Dropout + ReLU
 * optimizer: Adam /AdamW
