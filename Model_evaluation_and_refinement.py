from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def spliting(df):
    X = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
    Y = df[["price"]]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state= 0)
    return x_train, x_test, y_train, y_test


def cross_validation(df):
    lr = LinearRegression()
    X = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
    Y = df[["price"]]
    scores = cross_val_score(lr, X, Y, cv = 3)
    print("Mean score:", np.mean(scores))

    Yhat = cross_val_predict(lr, X, Y, cv = 3)

    fig, ax = plt.subplots()
    ax.scatter(Y, Yhat, edgecolors=(0, 0, 0))
    ax.plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.show()


def fitting_and_model_selection(df):
    x_train, x_test, y_train, y_test = spliting(df)
    Rsqu_test = []
    lr = LinearRegression()
    order = [1,2,3,4]
    for n in order:
        pr = PolynomialFeatures(degree = n)
        x_train_pr = pr.fit_transform(x_train[["horsepower", "curb-weight"]])
        x_test_pr = pr.fit_transform(x_test[["horsepower", "curb-weight"]])
        lr.fit(x_train_pr, y_train)
        Rsqu_test.append(lr.score(x_test_pr, y_test))

    print(Rsqu_test)


def ridge_regression(df):
    X = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
    Y = df[["price"]]
    alpha_values = [1, 10, 100, 1000, 10000]

    axl = sns.kdeplot(df["price"], color = 'r', label = "Actual value")
    plt.title("Ridge Regression")

    for alpha in alpha_values:
        RidgeModel = Ridge(alpha = alpha)
        RidgeModel.fit(X, Y)
        Yhat = RidgeModel.predict(X)
        print(alpha, ":", RidgeModel.score(X, Y))
        sns.kdeplot(np.concatenate(Yhat, axis = 0), color = 'b', label = alpha, ax = axl)

    plt.show()


# [0.628996   0.62899614 0.62899761 0.62915768 0.63049259 0.63715195 0.63852717 0.60657494]
def grid_search(df):
    X = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
    Y = df[["price"]]
    parameters = [{'alpha': [0.001, 0.01, 0.1, 10, 100, 1000, 10000, 100000],
                   'normalize': [True, False]}] # uses python dictionary
    RR = Ridge()
    Grid1 = GridSearchCV(RR, parameters, cv = 4)
    Grid1.fit(X, Y)
    print("best estimator", Grid1.best_estimator_)
    scores = Grid1.cv_results_
    print("Mean score grid search: ", scores['mean_test_score'])
    print(scores)

    for params, mean_test in zip(scores['params'], scores['mean_test_score']):
        print(params, "R^2 on test data:", mean_test)

