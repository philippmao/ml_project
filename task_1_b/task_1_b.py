from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import csv
import numpy as np
import math


def main():
    x = np.zeros((900,21))
    y = np.zeros(900)
    with open('train.csv') as trainfile:
        data = csv.reader(trainfile, delimiter=',')
        for row in data:
            if row[0] != 'Id':
                y[int(row[0])] = row[1]
                i = 0
                for x_i in row[2:7]:
                    x_i = float(x_i)
                    x[int(row[0])][i] = x_i
                    x[int(row[0])][i + 5] = math.pow(x_i, 2)
                    x[int(row[0])][i + 10] = math.exp(x_i)
                    x[int(row[0])][i + 15] = math.cos(x_i)
                    i += 1
                x[int(row[0])][20] = 1
    x_train = x[range(0, 850)]
    x_test = x[range(850, 900)]
    y_train = y[range(0, 850)]
    y_test = y[range(850, 900)]
    clf = LinearRegression()
    clf.fit(x_train, y_train)
    print(clf.coef_)
    y_predict = clf.predict(x_test)
    RMSE = mean_squared_error(y_test, y_predict) ** 0.5
    print(RMSE)
    with open('results.csv', 'w') as resultfile:
        for w in clf.coef_:
            w = str(w)
            resultfile.write(w)
            resultfile.write('\n')

if __name__ == "__main__":
    main()
