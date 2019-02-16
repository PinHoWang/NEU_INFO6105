
import pandas as pd

df = pd.read_csv('data.csv', decimal = ',')
print(df.head())
print(list(df))