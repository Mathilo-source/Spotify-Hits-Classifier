import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  
from IPython.display import display 

#loading the dataset
def load_dataset():
    df_audio=pd.read_csv('audio_features.csv')
    df_chart=pd.read_csv('chart_longevity_panel.csv')
    df_meta=pd.read_csv('track_metadata.csv')
    print("Datasets loaded successfully")
    return df_audio, df_chart, df_meta

#merging the data
def merge_data(df_audio, df_chart, df_meta):
    df = pd.merge(df_audio, df_chart, on='track_id', how='inner')
    df = pd.merge(df, df_meta, on='track_id', how='inner')
    print("Datasets merged successfully")
    return df

#inspecting the data
def inspect_data(df):
    print("Dataframe shape:", df.shape)
    print("Dataframe info:")
    df.info()
    print("Dataframe description:")
    print(df.describe())
    print("Dataframe head:")
    print(df.head())
    print("Dataframe tail:")
    print(df.tail())
    print("Dataframe columns:")
    print(df.columns)
    print("Dataframe missing values:")
    print(df.isnull().sum())
    print("Dataframe duplicates:")
    print(df.duplicated().sum())
    print("Dataframe unique values:")
    print(df.nunique())

#the start of the data cleaning process
    print("=" * 50)
    print("DATA CLEANING")
    print("=" * 50)

#changing the data types of columns
def change_release_date_type(df):
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    print("-------THE DATASET AFTER CONVERTING release_date TO DATETIME-------")
    df.info()
    return df

#cleaning the column names
def clean_colnames(df):
    print("-------THE COLUMN NAMES AFTER CLEANING-------")
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
    print(df.columns)
    return df

#inconsistencies in the data
def check_inconsistent_values(df):
    print("-------THE UNIQUE VALUES IN THE genre COLUMN-------")
    print(df['genre'].value_counts(dropna=False))
    print("-------THE UNIQUE VALUES IN THE mode COLUMN-------")
    print(df['mode'].value_counts(dropna=False))

# Standardize text values
def clean_text_values(df):
    print("-------CLEANING TEXT VALUES IN THE DATASET-------")
    text_columns = df.select_dtypes(
        include=['object', 'string']
    ).columns

    for col in text_columns:
        df[col] = df[col].str.strip()

    return df

#exporting the cleaned dataset
def export_cleaned_dataset(df):
    df.to_csv('clean_df.csv', index=False)
    print("Cleaned dataset exported successfully to 'clean_df.csv'")
#the main function to run the script
def main():
    df_audio, df_chart, df_meta = load_dataset()
    df = merge_data(df_audio, df_chart, df_meta)
    inspect_data(df)
    change_release_date_type(df)
    clean_colnames(df)
    check_inconsistent_values(df)
    df = clean_text_values(df)
    export_cleaned_dataset(df)
    return df

if __name__ == "__main__":
    df = main()