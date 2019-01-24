
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error

df = pd.read_csv('COTW.csv', decimal=',')

# Find out the sequence of correlation and use them as the features for 
# ploting degree vs error (training/testing) digram
gdp_corr_seq = (df.corr()['GDP ($ per capita)']).sort_values(ascending=False).drop('GDP ($ per capita)')

# print(df.isnull().sum())

for col in df.columns.values:
    if df[col].isnull().sum() == 0:
        continue
    if col == 'Climate':
        guess_values = df.groupby('Region')['Climate'].apply(lambda x: x.mode().max())
    else:
        guess_values = df.groupby('Region')[col].median()
    for region in df['Region'].unique():
        df[col].loc[(df[col].isnull())&(df['Region']==region)] = guess_values[region]

LE = LabelEncoder()
df['Regional_label'] = LE.fit_transform(df['Region'])
df['Climate_label'] = LE.fit_transform(df['Climate'])

train, test = train_test_split(df, test_size=0.3, shuffle=True)
# training_features = ['Population', 'Area (sq. mi.)',
#        'Pop. Density (per sq. mi.)', 'Coastline (coast/area ratio)',
#        'Net migration', 'Infant mortality (per 1000 births)',
#        'Literacy (%)', 'Phones (per 1000)',
#        'Arable (%)', 'Crops (%)', 'Other (%)', 'Birthrate',
#        'Deathrate', 'Agriculture', 'Industry', 'Service', 'Regional_label',
#        'Climate_label','Service']


trainPlot = []
testPlot = []
x = []
for hyper in range(2, gdp_corr_seq.size):
# for hyper in range(2, 3):
    training_features = gdp_corr_seq[0:hyper].index[:]
    # print(training_features)
    target = 'GDP ($ per capita)'
    train_X = train[training_features]
    train_Y = train[target]
    test_X = test[training_features]
    test_Y = test[target]

    model = LinearRegression()
    model.fit(train_X, train_Y) # Training the model
    train_pred_Y = model.predict(train_X)
    test_pred_Y = model.predict(test_X)
    train_pred_Y = pd.Series(train_pred_Y.clip(0, train_pred_Y.max()), index=train_Y.index)
    test_pred_Y = pd.Series(test_pred_Y.clip(0, test_pred_Y.max()), index=test_Y.index)


    rmse_train = np.sqrt(mean_squared_error(train_pred_Y, train_Y))
    msle_train = mean_squared_log_error(train_pred_Y, train_Y)
    rmse_test = np.sqrt(mean_squared_error(test_pred_Y, test_Y))
    msle_test = mean_squared_log_error(test_pred_Y, test_Y)

    # print('Features number: ',training_features.size)
    # # print('rmse_train:',rmse_train,'msle_train:',msle_train)
    # # print('rmse_test:',rmse_test,'msle_test:',msle_test)
    # print('rmse_train:',rmse_train)
    # print('rmse_test:',rmse_test)
    # print('')
    x.extend([hyper])
    trainPlot.extend([rmse_train])
    testPlot.extend([rmse_test])

plt.plot(x, trainPlot, 'r+')
plt.plot(x, testPlot, 'bo')
plt.show()


