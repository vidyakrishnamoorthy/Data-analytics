# descriptive statistics
# Groupby
# correlations
# statistical corrlelations: pearsons, heatmaps

import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import pandas as pd

#      value_counts
# fwd           120
# rwd            76
# 4wd             9
def descriptive_statistics(df):
    drive_wheels_counts = df["drive-wheels"].value_counts().to_frame()
    drive_wheels_counts.rename(columns = {"drive-wheels": "drive-wheels_value_counts"}, inplace= True)
    print(drive_wheels_counts)

    make_counts = df["make"].value_counts().to_frame()
    make_counts.rename(columns={"make": "make_value_counts"}, inplace=True)
    print(make_counts)

def descriptive_plots(df):
    sns.boxplot(x = "drive-wheels", y="price", data = df)
    plt.show()

    sns.boxplot(x = "body-style", y="price", data = df)
    plt.show()

    sns.scatterplot(x = "city-L/100km", y = "price", data = df)
    sns.lineplot(x = "city-L/100km", y = "price", data = df)
    plt.show()

    sns.scatterplot(x = "engine-size", y = "price", data = df)
    sns.lineplot(x = "engine-size", y = "price", data = df)
    plt.title("Engine size vs Price")
    plt.xlabel("Engine size")
    plt.ylabel("Price")
    plt.show()

def group_by_usage(df):
    df_test = df[["drive-wheels", "body-style", "price"]]
    df_grp = df_test.groupby(["drive-wheels", "body-style"], as_index = False).mean()
    print(df_grp)

    df_pivot = df_grp.pivot(index = "drive-wheels", columns = "body-style")
    print(df_pivot)

    plt.pcolor(df_pivot, cmap = "RdBu")
    plt.colorbar()
    plt.show()

# Pivot table
# body-style   convertible       hardtop     hatchback         sedan         wagon
# drive-wheels
# 4wd                  NaN           NaN   3802.000000  12647.333333   9095.750000
# fwd              11595.0   8249.000000   8396.387755   9467.561404   9997.333333
# rwd              23949.6  24202.714286  13583.210526  21711.833333  16994.222222

def correlation(df):
    sns.regplot(x="engine-size", y = "price", data = df)
    plt.ylim(0,)
    plt.show()

    sns.regplot(x = "highway-mpg", y = "price", data = df)
    plt.ylim(0,)
    plt.show()

    sns.regplot(x = "peak-rpm", y = "price", data = df)
    plt.ylim(0,)
    plt.show()

def correlation_statistics(df):
    pearson_coef, p_value = stats.pearsonr(df["horsepower"], df["price"])
    print(pearson_coef, p_value) # 0.703526657349445 6.119291389677244e-32 (meaning, strong correlation)

    print(df.corr())
    sns.heatmap(df.corr())
    plt.show()

def relation_between_categorical_chi_square(df):
    contingency_table = pd.crosstab(df["fuel-type"], df["aspiration"])
    print(contingency_table)

    expected_chi_square = stats.chi2_contingency(contingency_table, correction = True)
    print(expected_chi_square)
