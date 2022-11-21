import os
import requests
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error as mape

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('../Data/data.csv')


# stage 2/ 5
def my_func(n):

    model = LinearRegression()
    X = data[["rating"]] ** n
    y = data["salary"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    map = mape(y_test, y_pred)
    return round(map, 5)


# my_list = []
# for i in range(2,5):
#     my_list.append(my_func(i))
# print(min(my_list))

# Stage 3/5: Linear regression with many independent variables

X = data.drop(columns="salary")
y = data['salary']
#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
model = LinearRegression()
model.fit(X_train, y_train)
#print(*model.coef_, sep=",")


# stage 4/5: Test for multicollinearity and variables selection

corr_matrix = X.corr()
new_set = []
for x in X.columns:
    for x2 in X.columns:
        if x != x2 and corr_matrix[x][x2] > 0.2:
            new_set.append(x)
            new_set.append(x2)
my_set =[]
for el in new_set:
    if el not in my_set:
        my_set.append(el)
def best_pred(x, data, y, rep=None):
    X = data.drop(columns=x)
    model = LinearRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
    if rep=="0":
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred[y_pred < 0] = 0
        map = mape(y_test, y_pred)
        return round(map, 5)
    elif rep == 'median':

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred[y_pred < 0] = np.median(y_train)
        map = mape(y_test, y_pred)
        return round(map, 5)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        map = mape(y_test, y_pred)
        return round(map, 5)


#

p1 = best_pred(my_set[0], X, y)
p2 = best_pred(my_set[1], X, y)
p3 = best_pred(my_set[2], X, y)
p4 = best_pred(my_set[0:2], X, y)
p5 = best_pred(my_set[1:3], X, y)
p6 = best_pred(my_set[0:3:2], X, y)

m = [p1, p2, p3, p4, p5, p6]

#print(min(m))


# stage 5/5 : Deal with negative predictions


pred = best_pred(["age","experience"], X, y, "0")
pred2 =best_pred(["age","experience"], X, y, "median")
print(min(pred, pred2))



