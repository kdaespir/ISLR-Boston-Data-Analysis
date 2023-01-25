import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import statistics as st
import matplotlib.pyplot as plt


data = pd.read_csv("boston.csv")

x = data["rm"]
y = data['medv']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)

x_train = x_train.values
x_train = x_train.reshape(-1,1)

x_test = x_test.values
x_test = x_test.reshape(-1,1)

def boston_lr_rm_on_medv():

    lm = LinearRegression()
    lm.fit(x_train,y_train)

    test_pred = lm.predict(x_test)
    train_pred = lm.predict(x_train)
    

    train_mse = mean_squared_error(y_train,train_pred)
    test_mse = mean_squared_error(y_test,test_pred)
    print(train_mse)
    # the MSE for the training data is 304.123
    print(test_mse)
    # the MSE for the test data is 282.906

    # plt.scatter(x_test,y_test, color="red")
    # plt.plot(x_test, lm.predict(x_test), color="blue")
    # plt.title("Number of Rooms Regressed on Median Home Value")
    # plt.xlabel("Number of Rooms")
    # plt.ylabel("Median Home Value")
    # plt.show()

def boston_pr_rm_on_mdev(degrees):

    poly_feat = PolynomialFeatures(degree=degrees)
    x_poly = poly_feat.fit_transform(x_train)
    pr = LinearRegression()
    pr.fit(x_poly, y_train)
    x_poly_test = poly_feat.fit_transform(x_test)
    
    pr_train_pred = pr.predict(x_poly)
    pr_test_pred = pr.predict(x_poly_test)

    print(y_train)
    pr_train_mse = mean_squared_error(y_train, pr_train_pred)
    pr_test_mse = mean_squared_error(y_test, pr_test_pred)

    print(f'The training MSE for {degrees} is {pr_train_mse},\nThe testing MSE for {degrees} is {pr_test_mse}')
    print("Finished!!!")

    return [pr_test_mse, pr_train_mse]

def boston_loocv():
    lr = LinearRegression()
    cv = LeaveOneOut()

    tr_score = cross_val_score(lr, x_train, y_train, cv=cv, scoring="neg_mean_squared_error")
    tr_score = abs(tr_score)
    print(st.mean(tr_score))
    pass

def boston_kfcv():
    lr = LinearRegression()
    cv = KFold(10)

    tr_score = cross_val_score(lr, x_train, y_train, cv=cv, scoring="neg_mean_squared_error")
    tr_score = abs(tr_score)
    print(st.mean(tr_score))
    pass

if __name__ == "__main__":
    # Question 1 A
    boston_lr_rm_on_medv()
    boston_pr_rm_on_mdev(2)
    
    #Question 1 B

    flex = list(range(1,51))
    test_mse_scores = []
    train_mse_scores = []
    for item in range(1, 51):
        scores = boston_pr_rm_on_mdev(degrees=item)
        test_mse_scores += [scores[0]]
        train_mse_scores += [scores[1]]
    
    plt.scatter(flex, test_mse_scores, color="black")
    plt.scatter(flex, train_mse_scores, color="blue")
    plt.title("MSE Changes Per Flexibility Increase")
    plt.xlabel("Polynomial Degree")
    plt.show()

    #Question 1 C
    boston_loocv()

    # #Question 1 D
    boston_kfcv()
    pass
