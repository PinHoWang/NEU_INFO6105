

import matplotlib.pyplot as plt
import pandas as pd



df = pd.read_csv('2012-18_playerBoxScore.csv', decimal=',')
print(list(df))
print(df.isnull().sum().sum())
