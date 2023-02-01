import pandas as pd
from log_models import SelectionTable, PredictionLog
from file_map import FileMap
from collections import namedtuple
from datetime import timedelta
import os
from utilities import get_range_overlap, get_weighted_prediction

Range = namedtuple('Range', ['start', 'end'])


def analyze_event_detections(gt_file, prediction_csv, file_map: FileMap):
    pred_df = pd.read_csv(prediction_csv, parse_dates=['binary_start', 'binary_end'])
    pred_df = pred_df[pred_df["ground_truth_file"] == os.path.basename(gt_file)]

    table = SelectionTable(gt_file)
    binary_prediction_logs = file_map.gt_pred_map[gt_file]
    data = {
        "ground_truth_file": [],
        "binary_prediction_file": [],
        "multiclass_prediction_file": [],
        "binary_start": [],
        "binary_end": [],
        "binary_start_rel": [],
        "binary_end_rel": [],
        "ground_truth_start": [],
        "ground_truth_end": [],
        "ground_truth_start_rel": [],
        "ground_truth_end_rel": [],
        "multiclass_prediction": [],
        "ground_truth": [],
        "quality": [],
        "notes": [],
        "song": [],
        "call_type": [],
        "likely_sex": []
    }
    for entry in table.data:
        entry_start = table.start_date_time + timedelta(seconds=entry["start_t"])
        entry_end = table.start_date_time + timedelta(seconds=entry["end_t"])
        r1 = Range(start=entry_start, end=entry_end)
        data["ground_truth_file"].append(os.path.basename(gt_file))
        data["ground_truth"].append(entry["class_id"])
        data["ground_truth_start_rel"].append((r1.start - table.start_date_time).total_seconds())
        data["ground_truth_end_rel"].append((r1.end - table.start_date_time).total_seconds())
        data["ground_truth_start"].append(r1.start)
        data["ground_truth_end"].append(r1.end)

        pred = {
            "binary_prediction_file": "",
            "binary_start": "",
            "binary_end": "",
            "binary_start_rel": "",
            "binary_end_rel": "",
            "multiclass_prediction_file": "",
            "multiclass_prediction": "",
            "quality": "",
            "notes": "",
            "song": "",
            "call_type": "",
            "likely_sex": "",
        }
        found = False
        for idx, row in pred_df.iterrows():
            r2 = Range(start=row['binary_start'], end=row['binary_end'])
            overlap = get_range_overlap(r1, r2)
            if overlap > 0:
                found = True
                data["binary_prediction_file"].append(row["binary_prediction_file"])
                data["binary_start"].append(row["binary_start"])
                data["binary_end"].append(row["binary_end"])
                data["binary_start_rel"].append(row["binary_start_rel"])
                data["binary_end_rel"].append(row["binary_end_rel"])
                data["multiclass_prediction_file"].append(row["multiclass_prediction_file"])
                data["multiclass_prediction"].append(row["multiclass_prediction"])
                data["quality"].append(row["quality"])
                data["notes"].append(row["notes"])
                data["song"].append(row["song"])
                data["call_type"].append(row["call_type"])
                data["likely_sex"].append(row["likely_sex"])
                break

        if not found:
            data["binary_prediction_file"].append(pred["binary_prediction_file"])
            data["binary_start"].append(pred["binary_start"])
            data["binary_end"].append(pred["binary_end"])
            data["binary_start_rel"].append(pred["binary_start_rel"])
            data["binary_end_rel"].append(pred["binary_end_rel"])
            data["multiclass_prediction_file"].append(pred["multiclass_prediction_file"])
            data["multiclass_prediction"].append(pred["multiclass_prediction"])
            data["quality"].append(pred["quality"])
            data["notes"].append(pred["notes"])
            data["song"].append(pred["song"])
            data["call_type"].append(pred["call_type"])
            data["likely_sex"].append(pred["likely_sex"])

    return pd.DataFrame(data)





if __name__ == '__main__':
    dfs = []
    f_map = FileMap("results/file_map.json")
    pred_csv = "/home/alex/data/KARAN_ODOM/analysis/results/prediction_results.csv"
    for gt in list(f_map.gt_pred_map.keys()):
        gt_df = gt_pred_results = pd.DataFrame(analyze_event_detections(gt, pred_csv, f_map))\
            .sort_values(by=["binary_start"], ascending=True)\
            .reset_index()\
            .drop(columns=["index"])
        dfs.append(gt_df)

    df = pd.concat(dfs, axis=0)
    df.to_csv("results/ground_truth_analysis.csv", index=False)
