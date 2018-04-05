from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB

import csv
import numpy as np
import math


def main():
    x = np.zeros((2000, 16))
    y = np.zeros(2000)
    with open('train.csv') as trainfile:
        data = csv.reader(trainfile, delimiter=',')
        for row in data:
            if row[0] != 'Id':
                y[int(row[0])] = row[1]
                i = 0
                for item in row[2:18]:
                    x[int(row[0])][i] = item
                    i += 1
    lamdba_list = [0.1, 1, 10, 100, 1000]
    x_train = x[range(0, 1900)]
    x_test = x[range(1900, 2000)]
    y_train = y[range(0, 1900)]
    y_test = y[range(1900, 2000)]
    for a in lamdba_list:
        svc = SVC(C= a, kernel='linear')
        clf = GaussianNB()
        svc.fit(x_train, y_train)
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print("gaussian bayes", acc)
        y_predict = svc.predict(x_test)
        acc = accuracy_score(y_test, y_predict)
        print("svm", acc)
    #with open('results.csv', 'w') as resultfile:
    #    for w in clf.coef_:
    #        w = str(w)
    #        resultfile.write(w)
    #        resultfile.write('\n')

if __name__ == "__main__":
    main()
