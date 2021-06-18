from Importing_exporting_data import *
from Preprocessing import *
from Exploratory_data_analysis import *
from Model_development import *
from Model_evaluation_and_refinement import *

def main():
    print("Hello!")
    path = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
    df = read_from_path(path)
    save_path = "Data/imports-85.csv"
    save_df_csv(df, save_path)
    describe_df(df)

    # Data pre-processing
    df = dropping_missing(df)
    df = replacing_missing(df)
    df = data_formatting(df)
    df = data_normalization(df, "simple feature")
    df = data_normalization(df, "min-max")
    df = data_normalization(df, "z-score")
    df = binning(df)

    # Descriptive analysis
    categorical_to_quantitative(df)
    descriptive_statistics(df)
    descriptive_plots(df)
    group_by_usage(df)
    correlation(df)
    correlation_statistics(df)
    relation_between_categorical_chi_square(df)

    # Model development
    single_linear_regression(df)
    model_using_visualization(df)
    polynomial_regression(df)
    pipelines(df)
    predictions(df)
    spliting(df)
    cross_validation(df)
    fitting_and_model_selection(df)
    ridge_regression(df)
    grid_search(df)


main()