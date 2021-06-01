import numpy as np
import pandas as pd

def column_values(df):
    print(df["symboling"])
    print(df["body-style"])
    df["symboling"] = df["symboling"] + 1
    print(df["symboling"])

def dropping_missing(df):
    #print(df.describe())
    print(df.isna())
    df.dropna(subset = ["price"], axis=0, inplace = True)
    #print(df.describe())

    #print(df["price"])
    return df

def replacing_missing(df):
    print(df["price"][9])
    df = df.replace({"price": '?'}, True)
    print(df["price"][9])

    df = df.replace({"normalized-losses": '?'}, np.nan)
    df["normalized-losses"] = df["normalized-losses"].astype(float, errors = 'raise')

    mean_normalised_losses = df["normalized-losses"].mean(skipna = True)
    df["normalized-losses"] = df["normalized-losses"].replace(np.nan, mean_normalised_losses)

    df = df.replace({"peak-rpm": '?'}, np.nan)
    df["peak-rpm"] = df["peak-rpm"].astype(float, errors = 'raise')

    df = df.replace({"horsepower": '?'}, np.nan)
    df["horsepower"] = df["horsepower"].astype(float, errors = 'raise')
    mean_horsepower = df["horsepower"].mean(skipna = True)
    df["horsepower"] = df["horsepower"].replace(np.nan, mean_horsepower)

    return df

def data_formatting(df):
    df["city-mpg"] = 235/df["city-mpg"]
    print(df["city-mpg"].tail(5))
    df.rename(columns = {"city-mpg": "city-L/100km"}, inplace = True)
    print(df["city-L/100km"].tail(5))

    print(df["price"].tail(5))
    df["price"] = df["price"].astype(float, errors = 'raise')
    print(df["price"].tail(5))

    return df

def data_normalization(df, norm_type):
    print(df["length"].head(5))
    switcher = {
        "simple feature": df["length"]/df["length"].max(),
        "min-max": (df["length"] - df["length"].min())/(df["length"].max() - df["length"].min()),
        "z-score": (df["length"] - df["length"].mean())/df["length"].std()
    }
    df["length"] = switcher.get(norm_type)
    print(df["length"].head(5))

    return df

def binning(df):
    bins = np.linspace(min(df["price"]), max(df["price"]), 4)
    print(min(df["price"]))
    print(max(df["price"]))
    print(bins)
    group_names = ["Low", "Medium", "high"]
    df["price-binned"] = pd.cut(df["price"], bins, labels = group_names, include_lowest = True)
    print(df["price-binned"])

    return df

def categorical_to_quantitative(df):
    print(pd.get_dummies(df["fuel-type"]))

    print(df.columns)
