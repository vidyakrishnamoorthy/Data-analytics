from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error

def single_linear_regression(df):
    lm = LinearRegression()
    X = df[["highway-mpg"]]
    Y = df[["price"]]

    lm.fit(X, Y)
    Yhat = lm.predict(X)
    #print(numpy.concatenate(Yhat, axis = 0))
    print("MSE:", mean_squared_error(Y, Yhat))
    print("Rsquared", lm.score(X,Y))
    print("Prediction:", lm.predict(np.array(30.0).reshape(-1, 1))) # Predicted price: 13555.50706069
    print(lm.intercept_, lm.coef_) # [37758.48383499] [[-806.76589248]]

    Z = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
    lm.fit(Z, Y)
    Yhat = lm.predict(Z)
    #print(Yhat)
    print("MSE:", mean_squared_error(Y, Yhat))
    print("Rsquared", lm.score(Z,Y))
    print(lm.intercept_, lm.coef_) # [-1932.73053419] [[ -15.78505192    2.97034008  120.15672652 -205.25362564]]

    A = df[["horsepower", "engine-size"]]
    lm.fit(A, Y)
    Yhat = lm.predict(A)
    print("MSE:", mean_squared_error(Y, Yhat))
    print("Rsquared", lm.score(A,Y))
    print(lm.intercept_, lm.coef_)

# using regplot and residual plot to verify the linearity of the model
def model_using_visualization(df):
    sns.regplot(x = "highway-mpg", y = "price", data = df)
    plt.ylim(0,)
    plt.show()

    sns.residplot(x = df["highway-mpg"], y = df["price"]) # if residuals has a pattern, nonlinear
    plt.show()

    lm = LinearRegression()
    X = df[["highway-mpg"]]
    Y = df[["price"]]
    lm.fit(X, Y)
    Yhat = lm.predict(X)
    print("MSE:", mean_squared_error(Y, Yhat))
    print("Rsquared", lm.score(X,Y))

    axl = sns.kdeplot(df["price"], color = 'r', label = "Actual value")
    sns.kdeplot(np.concatenate(Yhat, axis = 0), color = 'b', label = "Fitted value", ax = axl)
    plt.show()

    Z = df[["horsepower", "curb-weight", "engine-size", "highway-mpg"]]
    lm.fit(Z, Y)
    Yhat = lm.predict(Z)
    print("MSE:", mean_squared_error(Y, Yhat))
    print("Rsquared", lm.score(Z,Y))

    axl = sns.kdeplot(df["price"], color = 'r', label = "Actual value")
    sns.kdeplot(np.concatenate(Yhat, axis = 0), color = 'b', label = "Fitted value", ax= axl)
    plt.show()

#           3         2
# -0.01152 x + 4.679 x - 414.8 x + 1.771e+04
def polynomial_regression(df):
    x = df["horsepower"]
    y = df["price"]
    f = np.polyfit(x,y,3)
    p = np.poly1d(f)
    print(p)

    pr = PolynomialFeatures(degree = 2, include_bias = False)
    x_poly = pr.fit_transform(df[["horsepower", "curb-weight"]])
    print("x_poly\n", x_poly)
    print("pr.fit_transform([[1,2]])\n", pr.fit_transform([[1,2]]))

    SCALE = StandardScaler()
    SCALE.fit(df[["horsepower", "highway-mpg"]])
    x_scale = SCALE.transform(df[["horsepower", "highway-mpg"]])

def pipelines(df):
    input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(degree = 2)),
             ('mode', LinearRegression())] # tuple: ('name_of_estimator', model_constructor)
    pipe = Pipeline(input)
    X = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]
    Y = df["price"]

    pipe.fit(X, Y)
    Yhat = pipe.predict(df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
    print("MSE:", mean_squared_error(Y, Yhat))
    print("Rsquared", pipe.score(X, Y))

    axl = sns.kdeplot(df["price"], color='r', label="Actual value")
    sns.kdeplot(Yhat, color='b', label="Fitted value", ax=axl)
    plt.show()


def predictions(df):
    lm = LinearRegression()
    X = df[["highway-mpg"]]
    Y = df[["price"]]

    lm.fit(X, Y)
    Yhat = lm.predict(X)
    print("MSE:", mean_squared_error(Y, Yhat))
    print("Rsquared", lm.score(X,Y))
    print(lm.intercept_, lm.coef_) # [37758.48383499] [[-806.76589248]]

    print("Prediction:", lm.predict(np.array(30.0).reshape(-1, 1))) # Predicted price: 13555.50706069
    new_input = np.arange(1, 101, 1).reshape(-1,1)
    print("Prediction:", lm.predict(new_input)) # Predicted price: 13555.50706069
