import pandas as pd

#loading the dataset
def load_dataset():
    df = pd.read_csv("feature_engineered_df.csv", parse_dates=["release_date"])
    print("Data loaded successfully")

    if "is_viral_hit" not in df.columns:
        raise ValueError("The dataset must contain the target column 'is_viral_hit'.")

    print(df["is_viral_hit"].value_counts(dropna=False).sort_index())
    print(df.info())
    return df

#selecting features and target variable
def select_features(df):
    target= "is_viral_hit"
    leakagecols= ["peak_chart_rank","weeks_on_top_100","total_streams_mil","longevity_score","artist_track_count_in_dataset","genre_track_count_in_dataset",]
    identifier_columns = ["track_id", "track_title", "artist_name", "release_date"]
    numeric_features = [
        "tempo_bpm",
        "danceability",
        "energy",
        "valence",
        "acousticness",
        "liveness",
        "duration_ms",
        "release_year",
        "release_month",
        "release_day_of_week",
        "is_explicit",
    ]
    categorical_features = ["musical_key", "mode", "genre"]








def main():
    df = load_dataset()

if __name__ == "__main__":
    main()  