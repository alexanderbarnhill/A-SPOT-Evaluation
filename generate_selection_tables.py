import pandas as pd
import os


def convert_to_selection_table(df, output_directory):
    gt_files = df["ground_truth_file"].unique()

    for gt in gt_files:
        table_selection = df[df["ground_truth_file"] == gt]

        table = {
            "Selection": [i for i in range(1, len(table_selection) + 1)],
            "View": ["Spectrogram 1" for _ in range(len(table_selection))],
            "Channel": [1 for _ in range(len(table_selection)) ],
            "Begin Time (s)": table_selection["binary_start_rel"],
            "End Time (s)": table_selection["binary_end_rel"],
            "prediction": table_selection["multiclass_prediction"],
            "ground_truth": table_selection["ground_truth"],
            "quality": table_selection["quality"],
            "notes": table_selection["notes"],
            "song": table_selection["song"],
            "call_type": table_selection["call_type"],
            "likely_sex": table_selection["likely_sex"],
        }
        t_df = pd.DataFrame(table)
        t_df.to_csv(os.path.join(output_directory, gt.replace('.txt', 'predictions.txt')), sep="\t", index=False)


if __name__ == '__main__':
    file = "/home/alex/data/KARAN_ODOM/analysis/results/past_sunrise_prediction_results.csv"
    output = "/home/alex/data/KARAN_ODOM/analysis/results"
    f = pd.read_csv(file)
    convert_to_selection_table(f, output)