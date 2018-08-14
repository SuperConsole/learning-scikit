import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

data=pd.read_csv('consumerPrices_tree.csv')
count=data['大都市圏分類'].value_counts()
X = data.drop(['都道府県', '大都市圏分類'], axis=1)
Y = data['大都市圏分類']

X_train,X_test,Y_train,Y_test = train_test_split(X, Y,random_state=0, test_size=0.3)

model = DecisionTreeClassifier(max_depth=3, random_state=0)
model.fit(X_train, Y_train)

print(metrics.accuracy_score(Y_test, model.predict(X_test)))
print(metrics.accuracy_score(Y_train, model.predict(X_train)))
