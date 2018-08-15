import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#元データのプロット・整形
X, Y=make_moons(n_samples=1000, noise=0.3, random_state=0)
data=pd.DataFrame({'X0':X[:, 0], 'X1':X[:, 1], 'Y':Y})

#学習
mlpModel = MLPClassifier(solver='lbfgs', random_state=0, hidden_layer_sizes=(20,))
mlpModel.fit(X, Y)

#データ付与
predicted = pd.DataFrame({'mlpPredicted':mlpModel.predict(X)})
data_predicted = pd.concat([data, predicted], axis =1)

#分類結果プロット
plt.scatter(data_predicted['X0'],data_predicted['X1'], c=data_predicted['Y'])
plt.scatter(data_predicted['X0'],data_predicted['X1'], c=data_predicted['mlpPredicted'], marker='x')
print(accuracy_score(Y, mlpModel.predict(X)))

plt.show()
