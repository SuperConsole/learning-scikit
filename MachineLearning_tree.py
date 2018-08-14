import pandas as pd
from sklearn.tree import DecisionTreeClassifier
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

#訓練データとテストデータ分割・学習
X_train, X_test, Y_train, Y_test = train_test_split(data_tmp_X, data_tmp_Y,random_state=0, test_size=0.3)
treeModel=DecisionTreeClassifier(max_depth=2, random_state=0)
treeModel.fit(X_train, Y_train)

#テスト
print('train: '+str(metrics.accuracy_score(Y_train, treeModel.predict(X_train))))
print('test: '+str(metrics.accuracy_score(Y_test, treeModel.predict(X_test))))

#重要要因抽出
importance=pd.DataFrame({'変数名':data_tmp_X.columns, '重要度':treeModel.feature_importances_})
print(importance[importance['重要度'] !=0])
