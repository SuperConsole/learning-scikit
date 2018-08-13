import pandas as pd
import matplotlib.pyplot
from sklearn.cluster import KMeans

#データの抽出
sample_data = pd.read_csv("consumerPrices.csv")
data=sample_data[['都道府県','食料','水道光熱費','保険医療']]

#KMeans法によるクラスタリング
model=KMeans(n_clusters=4,random_state=0)
data_X = data[['食料', '水道光熱費','保険医療']]
model.fit(data_X)

#結果結合
y1 = model.labels_
data_results=data.copy()
data_results['分類結果']=y1

#プロット
plt.scatter(data_results['食料'],data_results['水道光熱費'],c=data_results['分類結果'])
plt.show()
