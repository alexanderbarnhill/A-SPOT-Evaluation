import pandas as pd
import numpy as np
from datetime import datetime


def filter_by_sunrise(df, sunrise_times_csv):
    s_df = pd.read_excel(sunrise_times_csv, parse_dates=["sunrise"])
    df["sr"] = df["binary_start"].dt.strftime("%H%M%S")
    s_df["sr"] = s_df["sunrise"].dt.strftime("%H%M%S")
    result_df = []
    file_names = s_df["Annotation_Filename"]
    for i in s_df.itertuples():
        target_df = df[(df["ground_truth_file"] == i.Annotation_Filename) & (df["sr"] > i.sr)]
        result_df.append(target_df)
    result_df = pd.concat(result_df, axis=0)
    return result_df

def analyze(csv, sunrise_times_csv):
    df = pd.read_csv(csv, parse_dates=["binary_start"])
    past_sunrise_df = filter_by_sunrise(df, sunrise_times_csv)
    past_sunrise_df.to_csv(csv.replace("prediction_results.csv", "past_sunrise_prediction_results.csv"), index=False)
    positives = past_sunrise_df[(~past_sunrise_df["ground_truth_start"].isna()) & (past_sunrise_df["multiclass_prediction"] != 'noise')]
    negatives = past_sunrise_df[past_sunrise_df["ground_truth_start"].isna() & (past_sunrise_df["multiclass_prediction"] != 'noise')]
    tps = positives[positives["multiclass_prediction"] == positives["ground_truth"]]
    print(f"Total Predictions: {len(df)}")
    print(f"Predictions where sex is unknown: {len(positives[positives['ground_truth'] == 'u'])}")
    print(f"False Positives: {len(negatives)}")
    print(f"Predictions matching an annotated signal: {len(positives)}")
    print(f"True Positives: {len(tps)}")


if __name__ == '__main__':
    file = "/home/alex/data/KARAN_ODOM/analysis/results/prediction_results.csv"
    sunrise_file = "/home/alex/data/KARAN_ODOM/evaluation_soundfile_sunrise_times.xlsx"
    analyze(file, sunrise_file)