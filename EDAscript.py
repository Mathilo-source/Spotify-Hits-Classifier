import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pathlib

# Folder for saving EDA plots
OUTPUT_DIR = pathlib.Path("EDA_Outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Function for saving plots
def save_plot(filename, subfolder):
    folder = OUTPUT_DIR / subfolder
    folder.mkdir(parents=True, exist_ok=True)

    plt.savefig(folder / filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {folder / filename}")
    plt.close()

#loading the dataset
def load_dataset():
    df=pd.read_csv('clean_df.csv',parse_dates=['release_date'])
    print("Dataset loaded successfully")
    return df


# Inspecting the data
def inspect_data(df):
    print("=" * 50)
    print("DATASET INSPECTION")
    print("=" * 50)

    print("\nDataframe shape:")
    print(df.shape)

    print("\nDataframe information:")
    df.info()

    print("\nDataframe head:")
    print(df.head())

    print("\nDataframe tail:")
    print(df.tail())

    print("\nDataframe columns:")
    print(df.columns.tolist())

    print("\nDataframe data types:")
    print(df.dtypes)

    print("\nDataframe description:")
    print(df.describe().T)


# Checking data quality
def check_data_quality(df):
    print("\n" + "=" * 50)
    print("DATA QUALITY CHECKS")
    print("=" * 50)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print("\nTotal missing values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nDuplicate track IDs:")
    print(df['track_id'].duplicated().sum())

    print("\nUnique values per column:")
    print(df.nunique())


# Analysing categorical variables
def analyze_categorical_data(df):
    print("\n" + "=" * 50)
    print("CATEGORICAL DATA ANALYSIS")
    print("=" * 50)

    categorical_columns = [
        'genre',
        'artist_name',
        'musical_key',
        'mode',
        'is_explicit',
        'is_viral_hit'
    ]

    for col in categorical_columns:
        print(f"\nValue counts for {col}:")
        print(df[col].value_counts(dropna=False))

    print("\nGenre percentages:")
    print(
        (df['genre'].value_counts(normalize=True) * 100)
        .round(2)
    )


# Analysing numerical variables
def analyze_numerical_data(df):
    print("\n" + "=" * 50)
    print("NUMERICAL DATA ANALYSIS")
    print("=" * 50)

    numerical_columns = [
        'tempo_bpm',
        'danceability',
        'energy',
        'valence',
        'acousticness',
        'liveness',
        'peak_chart_rank',
        'weeks_on_top_100',
        'total_streams_mil',
        'longevity_score',
        'duration_ms'
    ]

    print("\nNumerical summary:")
    print(df[numerical_columns].describe().T)

    print("\nMedians:")
    print(df[numerical_columns].median())

    print("\nStandard deviations:")
    print(df[numerical_columns].std())


# Checking potential outliers
def check_outliers(df):
    print("\n" + "=" * 50)
    print("OUTLIER ANALYSIS")
    print("=" * 50)

    numerical_columns = [
        'tempo_bpm',
        'danceability',
        'energy',
        'valence',
        'acousticness',
        'liveness',
        'peak_chart_rank',
        'weeks_on_top_100',
        'total_streams_mil',
        'longevity_score',
        'duration_ms'
    ]

    for col in numerical_columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[
            (df[col] < lower_bound) |
            (df[col] > upper_bound)
        ]

        print(f"\nPotential outliers in {col}:")
        print("Number of potential outliers:", len(outliers))
        print("Lower bound:", round(lower_bound, 3))
        print("Upper bound:", round(upper_bound, 3))


# Visualizing numerical distributions
def plot_distributions(df):
    print("\n" + "=" * 50)
    print("NUMERICAL DISTRIBUTIONS")
    print("=" * 50)

    numerical_columns = [
        'tempo_bpm',
        'danceability',
        'energy',
        'valence',
        'acousticness',
        'liveness',
        'total_streams_mil',
        'longevity_score',
        'duration_ms'
    ]

    for col in numerical_columns:
        plt.figure(figsize=(8, 4))

        sns.histplot(
            data=df,
            x=col,
            bins=30,
            kde=True
        )

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.tight_layout()
        save_plot(
            filename=f"{col}.png",
            subfolder="numerical_distributions"
        )


# Visualizing categorical distributions
def plot_categorical_data(df):
    print("\n" + "=" * 50)
    print("CATEGORICAL DISTRIBUTIONS")
    print("=" * 50)

    plt.figure(figsize=(10, 6))

    sns.countplot(
        data=df,
        y='genre',
        order=df['genre'].value_counts().index
    )

    plt.title("Number of Tracks by Genre")
    plt.xlabel("Number of Tracks")
    plt.ylabel("Genre")
    plt.tight_layout()
    save_plot(
        filename="genre_distribution.png",
        subfolder="categorical_distributions"
    )

    plt.figure(figsize=(7, 4))

    sns.countplot(data=df, x='is_viral_hit')

    plt.title("Viral vs Non-Viral Tracks")
    plt.xlabel("Viral Hit (0 = No, 1 = Yes)")
    plt.ylabel("Number of Tracks")
    plt.tight_layout()
    save_plot(
        filename="viral_hits.png",
        subfolder="categorical_distributions"
    )

    plt.figure(figsize=(7, 4))

    sns.countplot(data=df, x='is_explicit')

    plt.title("Explicit vs Non-Explicit Tracks")
    plt.xlabel("Explicit (0 = No, 1 = Yes)")
    plt.ylabel("Number of Tracks")
    plt.tight_layout()
    save_plot(
        filename="explicit_tracks.png",
        subfolder="categorical_distributions"
    )


# Correlation analysis
def analyze_correlations(df):
    print("\n" + "=" * 50)
    print("CORRELATION ANALYSIS")
    print("=" * 50)

    numerical_columns = [
        'tempo_bpm',
        'danceability',
        'energy',
        'valence',
        'acousticness',
        'liveness',
        'peak_chart_rank',
        'weeks_on_top_100',
        'total_streams_mil',
        'is_viral_hit',
        'longevity_score',
        'duration_ms'
    ]

    corr = df[numerical_columns].corr()

    print("\nCorrelation with longevity_score:")
    print(
        corr['longevity_score']
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 9))

    sns.heatmap(
        corr,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f'
    )

    plt.title("Correlation Heatmap")
    plt.tight_layout()
    save_plot(
        filename="correlation_heatmap.png",
        subfolder="correlation_analysis"
    )


# Comparing viral and non-viral tracks
def compare_viral_tracks(df):
    print("\n" + "=" * 50)
    print("VIRAL TRACK COMPARISON")
    print("=" * 50)

    print("\nAverage streams by viral status:")
    print(
        df.groupby('is_viral_hit')['total_streams_mil']
        .agg(['count', 'mean', 'median', 'std'])
        .round(2)
    )

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x='is_viral_hit',
        y='total_streams_mil'
    )

    plt.title("Total Streams: Viral vs Non-Viral Tracks")
    plt.xlabel("Viral Hit (0 = No, 1 = Yes)")
    plt.ylabel("Total Streams (Millions)")
    plt.tight_layout()
    save_plot(
        filename="total_streams_viral_comparison.png",
        subfolder="viral_comparison"
    )

    plt.figure(figsize=(8, 5))

    sns.boxplot(
        data=df,
        x='is_viral_hit',
        y='longevity_score'
    )

    plt.title("Longevity Score: Viral vs Non-Viral Tracks")
    plt.xlabel("Viral Hit (0 = No, 1 = Yes)")
    plt.ylabel("Longevity Score")
    plt.tight_layout()
    save_plot(
        filename="longevity_viral_comparison.png",
        subfolder="viral_comparison"
    )


def main():
    df = load_dataset()
    inspect_data(df)
    check_data_quality(df)
    analyze_categorical_data(df)
    analyze_numerical_data(df)
    check_outliers(df)
    plot_distributions(df)
    plot_categorical_data(df)
    analyze_correlations(df)
    compare_viral_tracks(df)

if __name__ == "__main__":
    main()