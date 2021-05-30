from Importing_exporting_data import *
from Preprocessing import *

def main():
    print("Hello!")
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
    df = read_from_path(path)
    save_path = "Data/imports-85.csv"
    save_df_csv(df, save_path)
    #column_values(df)
    describe_df(df)
    df = dropping_missing(df)
    df = replacing_missing(df)
    df = data_formatting(df)
    #df = data_normalization(df, "simple feature")
    #df = data_normalization(df, "min-max")
    df = data_normalization(df, "z-score")
    df = binning(df)
    df = categorical_to_quantitative(df)


main()