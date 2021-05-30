import pandas as pd
import requests
import io
# read the online file by the URL provided above, and assign it to variable "df"

#df = pd.read_csv(filepath_or_buffer = path, header = None)
def read_from_path(path):
    df = pd.read_csv(io.StringIO(requests.get(path).content.decode('utf-8')), header = None)

    headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration",
               "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base",
               "length", "width", "height", "curb-weight", "engine-type",
               "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke",
               "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]
    df.columns = headers
    return df

def describe_df(df):
    print(df.head(10))
    print(df.tail(10))
    print(df.dtypes)
    print(df.describe())
    print(df.describe(include="all"))
    print(df.info)

def save_df_csv(df, path):
    df.to_csv(path) # save data as CSV


