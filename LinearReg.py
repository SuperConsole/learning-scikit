import pandas as pd
from sklearn.linear_model import LinearRegression

#データ整形・抽出
data=pd.read_csv('monthlyRent.csv')
X=data.drop(['都道府県','家賃'],axis=1)
Y=data['家賃']

#多重回帰
model=LinearRegression()
model.fit(X,Y)

#テスト
X_pred=pd.DataFrame([[4000,50000,100000,1000,14000,140]])
print(model.predict(X_pred))
