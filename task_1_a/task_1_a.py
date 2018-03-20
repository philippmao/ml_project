from sklearn.linear_model import Ridge
import csv
import numpy as np
import math


def main():
    x = np.zeros((500,10))
    y = np.zeros(500)
    with open('train.csv') as trainfile:
        data = csv.reader(trainfile, delimiter=',')
        for row in data:
            if row[0] != 'Id':
                y[int(row[0])] = row[1]
                i = 0
                for item in row[2:12]:
                    x[int(row[0])][i] = item
                    i += 1
    lamdba_list = [0.1, 1, 10, 100, 1000]
    # split data into 10 chunks for 10-fold cross validation
    # first chunk at indices 0-49, 50-99, 100-149  so on up to 450-499
    test_indices = list(range(0, 500))
    # dict mapping lambda to average result
    result_dict = dict()
    # list of ten measurements
    tmp_measurements = list(range(0,10))

    for lmdba in lamdba_list:
        for i in range(1,11):
            average = regr_average(x,y,i,lmdba)
            tmp_measurements[i-1] = average
        result_dict[lmdba] = sum(tmp_measurements)/len(tmp_measurements)

    print(result_dict)


def regr_average(x, y, cross_val_index, lamdba):
    indices= range(50*cross_val_index-50, 50*cross_val_index)
    print(indices)
    indices_list = list(indices)
    x_test = np.zeros((450, 10))
    y_test = np.zeros(450)
    counter = 0
    for i in range(0, 500):
        if i not in indices_list:
            x_test[counter] = x[i]
            y_test[counter] = y[i]
            counter = counter + 1
    clf = Ridge(alpha=lamdba)
    clf.fit(x_test,y_test)
    predicted_results = clf.predict(x[indices])
    RMSE = compute_RMSE(predicted_results, y[indices])
    return RMSE

def compute_RMSE(predicted_results, acutal_results):
    n = len(predicted_results)
    sum = 0
    for i in range(0, 50):
        sum = sum + (predicted_results[i]- acutal_results[i])*(predicted_results[i]- acutal_results[i])
    return math.sqrt(sum/n)

if __name__ == "__main__":
    main()
