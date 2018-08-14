import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

data=pd.read_csv('consumerPrices_tree.csv');

X = data.drop(['都道府県','大都市圏分類'],axis=1)
Y = data['大都市圏分類']
X_train,X_test,Y_train,Y_test = train_test_split(X, Y,random_state=0, test_size=0.3)

#DecisionTreeClassifier
treeModel = DecisionTreeClassifier(max_depth=3, random_state=0)
treeModel.fit(X_train, Y_train)

#LogisticRegression
logisticModel = LogisticRegression()
logisticModel.fit(X_train, Y_train)

#Output
print(data)

#Test
s1="Tree: "+str(metrics.accuracy_score(Y_test, treeModel.predict(X_test)))
s1_t="Tree_train: "+str(metrics.accuracy_score(Y_train, treeModel.predict(X_train)))

s2="Logis: "+str(metrics.accuracy_score(Y_test,logisticModel.predict(X_test)))
s2_t="Logis_train: "+str(metrics.accuracy_score(Y_train,logisticModel.predict(X_train)))
print()
print(s1_t)
print(s1)
print(s2_t)
print(s2)
