import pandas as pd

#loading the cleaned dataset
def load_dataset():
    df = pd.read_csv("clean_df.csv", parse_dates=["release_date"])
    print("Data loaded successfully")
    return df

#inspecting the data
def inspect_data(df):
    """Display the dataset structure and check basic data quality."""
    print("\n" + "=" * 60)
    print("PRE-FEATURE-ENGINEERING INSPECTION")
    print("=" * 60)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    print(f"\nDuplicate rows: {df.duplicated().sum()}")

#creating new date features
def create_date_features(df):
    """Derive calendar features from the release date."""
    #df = df.copy()

    df["release_year"] = df["release_date"].dt.year
    df["release_month"] = df["release_date"].dt.month
    df["release_day_of_week"] = df["release_date"].dt.dayofweek
    df["release_decade"] = (df["release_year"] // 10) * 10

    print("\nDate features created:")
    print("- release_year")
    print("- release_month")
    print("- release_day_of_week (Monday=0, Sunday=6)")
    print("- release_decade")

    return df

#creating the mins duration column
def create_duration_features(df):
    """Convert duration from milliseconds to minutes."""
    #df = df.copy()

    df["duration_min"] = df["duration_ms"] / 60000

    print("\nDuration feature created: duration_min")
    return df

#creating a categorical tempo feature
def create_tempo_features(df):
    """Create an interpretable tempo category using illustrative boundaries."""
    #df = df.copy()

    bins = [0, 80, 120, 160, float("inf")]
    labels = ["Slow", "Moderate", "Fast", "Very fast"]

    df["tempo_group"] = pd.cut(
        df["tempo_bpm"],
        bins=bins,
        labels=labels,
        right=False
    )

    print("\nTempo feature created: tempo_group")
    return df

#how many tracks per artist
def create_artist_features(df):
    """
    Count how many rows/tracks each artist has in this dataset.
    """
    #df = df.copy()

    artist_counts = df["artist_name"].value_counts()
    df["artist_track_count_in_dataset"] = df["artist_name"].map(artist_counts)

    print("\nArtist feature created: artist_track_count_in_dataset")
    return df

#how many tracks per genre
def create_genre_features(df):
    """Add the number of dataset rows belonging to each genre."""
    #df = df.copy()

    genre_counts = df["genre"].value_counts()
    df["genre_track_count_in_dataset"] = df["genre"].map(genre_counts)

    print("\nGenre feature created: genre_track_count_in_dataset")
    return df

#inspecting the new features
def review_features(df):
    """Review new columns and check the output after feature engineering."""
    new_features = [
        "release_year",
        "release_month",
        "release_day_of_week",
        "release_decade",
        "duration_min",
        "tempo_group",
        "artist_track_count_in_dataset",
        "genre_track_count_in_dataset",
    ]

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING REVIEW")
    print("=" * 60)

    print("\nNew feature preview:")
    print(df[new_features].head(10).to_string(index=False))

    print("\nNew feature data types:")
    print(df[new_features].dtypes)

    print("\nMissing values in new features:")
    print(df[new_features].isnull().sum())

    print(f"\nFinal dataset shape: {df.shape}")
    return df

#save the new dataset with the new features
def save_dataset(df, output_path="feature_engineered_df.csv"):
    df.to_csv(output_path, index=False)
    print(f"Feature-engineered dataset saved to: {output_path}")

def main():
    df=load_dataset()
    inspect_data(df)
    df=create_date_features(df)
    df=create_duration_features(df)
    df=create_tempo_features(df)
    df=create_artist_features(df)
    df=create_genre_features(df)
    df=review_features(df)
    save_dataset(df)

if __name__ == "__main__":
    main()  