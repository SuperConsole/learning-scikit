import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

data=pd.read_csv('ConsumerExpenditures.csv')

#データの整形
data_tmp = data.copy()
del data_tmp['Date']
del data_tmp['消費支出']
data_tmp_X = data_tmp.copy()
del data_tmp_X['消費支出増加フラグ']
data_tmp_Y = data_tmp['消費支出増加フラグ']

#訓練データとテストデータの分割・学習
X_train, X_test, Y_train, Y_test=train_test_split(data_tmp_X, data_tmp_Y,random_state=0, test_size=0.3)
modelLogis=LogisticRegression()
modelLogis.fit(X_train, Y_train)

#テスト
print('train: '+str(metrics.accuracy_score(Y_train, modelLogis.predict(X_train))))
print('test: '+str(metrics.accuracy_score(Y_test, modelLogis.predict(X_test))))

#係数(重要度合)算出
Coef = pd.DataFrame({'変数名':data_tmp_X.columns, '係数':modelLogis.coef_[0]})
print(Coef.sort_values('係数'))
