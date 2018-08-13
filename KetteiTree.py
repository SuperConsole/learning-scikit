import pandas as pd
data=pd.read_csv('consumerPrices_tree.csv')
count=data['大都市圏分類'].value_counts()
print(count)
