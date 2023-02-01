from collections import namedtuple

import pandas as pd
import os
from mapping import get_files
from log_models import PredictionLog, SelectionTable
from datetime import datetime, timedelta
from file_map import FileMap
from utilities import get_range_overlap, time_overlap

Range = namedtuple('Range', ['start', 'end'])

def get_gt_from_binary_log(log: PredictionLog, file_map: FileMap):
    for (k, v) in file_map.gt_pred_map.items():
        if log.log_file in v.keys():
            return SelectionTable(k)
    return None


def raw_prediction_to_csv(log: PredictionLog, file_map: FileMap):

    table = get_gt_from_binary_log(log, file_map)
    if table is None:
        return
    data = {
        "log_file": [],
        "audio_file": [],
        "gt_file": [],
        "rel_time_start": [],
        "rel_time_end": [],
        "abs_audio_time_start": [],
        "abs_audio_time_end": [],
        "abs_gt_time_start": [],
        "abs_gt_time_end": [],
        "rel_gt_time_start": [],
        "rel_gt_time_end": [],
        "pred": [],
        "class_id": [],
        "prob": []
    }

    for d in log.data:
        data["log_file"].append(log.log_file)
        data["audio_file"].append(log.audio_file)
        data["gt_file"].append(table.file_path)
        data["rel_time_start"].append(d['start_t'])
        data["rel_time_end"].append(d['end_t'])
        data["pred"].append(d['pred'])
        data["class_id"].append(d['class_id'])
        data["prob"].append(d['prob'])
        data["abs_audio_time_start"].append(log.start_date_time + timedelta(seconds=d['start_t']))
        data["abs_audio_time_end"].append(log.start_date_time + timedelta(seconds=d['end_t']))
        abs_gt_time_start = table.start_date_time + timedelta(seconds=d['start_t'] + (log.start_date_time - table.start_date_time).total_seconds())
        abs_gt_time_end = table.start_date_time + timedelta(seconds=d['end_t'] + (log.start_date_time - table.start_date_time).total_seconds())
        data["abs_gt_time_start"].append(abs_gt_time_start)

        data["abs_gt_time_end"].append(abs_gt_time_end)
        data["rel_gt_time_start"].append((abs_gt_time_start - table.start_date_time).total_seconds())
        data["rel_gt_time_end"].append((abs_gt_time_end - table.start_date_time).total_seconds())

    df = pd.DataFrame(data)
    df.to_csv(f"results/raw_{os.path.basename(log.log_file.replace('.log', '.csv'))}", index=False)


def positive_prediction_to_csv(log: PredictionLog, file_map: FileMap):

    table = get_gt_from_binary_log(log, file_map)
    if table is None:
        return
    data = {
        "log_file": [],
        "audio_file": [],
        "gt_file": [],
        "rel_time_start": [],
        "rel_time_end": [],
        "abs_audio_time_start": [],
        "abs_audio_time_end": [],
        "abs_gt_time_start": [],
        "abs_gt_time_end": [],
        "rel_gt_time_start": [],
        "rel_gt_time_end": [],
        "class_id": [],
    }

    for d in log.group():
        data["log_file"].append(log.log_file)
        data["audio_file"].append(log.audio_file)
        data["gt_file"].append(table.file_path)
        data["rel_time_start"].append(d['start_t'])
        data["rel_time_end"].append(d['end_t'])
        data["class_id"].append(d['class_id'])
        data["abs_audio_time_start"].append(log.start_date_time + timedelta(seconds=d['start_t']))
        data["abs_audio_time_end"].append(log.start_date_time + timedelta(seconds=d['end_t']))
        abs_gt_time_start = table.start_date_time + timedelta(seconds=d['start_t'] + (log.start_date_time - table.start_date_time).total_seconds())
        abs_gt_time_end = table.start_date_time + timedelta(seconds=d['end_t'] + (log.start_date_time - table.start_date_time).total_seconds())
        data["abs_gt_time_start"].append(abs_gt_time_start)

        data["abs_gt_time_end"].append(abs_gt_time_end)
        data["rel_gt_time_start"].append((abs_gt_time_start - table.start_date_time).total_seconds())
        data["rel_gt_time_end"].append((abs_gt_time_end - table.start_date_time).total_seconds())

    df = pd.DataFrame(data)
    df.to_csv(f"results/positive_{os.path.basename(log.log_file.replace('.log', '.csv'))}", index=False)


def count(folder, file_map: FileMap):
    files = get_files(folder, "log")
    predictions = []
    for file in files:
        log = PredictionLog(file)
        raw_prediction_to_csv(log, file_map)
        positive_prediction_to_csv(log, file_map)
        preds = log.group()
        predictions += preds

    print(len(predictions))


def analyze_positives(results_folder):
    df = pd.concat([pd.read_csv(os.path.join(results_folder, f)) for f in os.listdir(results_folder) if "positive_" in f], axis=0)
    gt_dfs = []
    for gt_file in list(df["gt_file"].unique()):
        gt_df = pd.read_csv(gt_file, sep="\t")
        gt_df["gt_file"] = gt_file
        gt_dfs.append(gt_df)
    gt_dfs = pd.concat(gt_dfs, axis=0)
    overlaps = {
        "sel_table": [],
        "rel_gt_start": [],
        "rel_gt_end": [],
        "rel_pred_start": [],
        "rel_pred_end": []
    }

    for row in df.itertuples():
        row_gt = gt_dfs[gt_dfs["gt_file"] == row.gt_file]
        pred_range = Range(start=row.rel_gt_time_start, end=row.rel_gt_time_end)
        #pred_range = [row.rel_gt_time_start, row.rel_gt_time_end]
        for gt_row in row_gt.itertuples():
            gt_range = Range(start=gt_row[4], end=gt_row[5])
            # gt_range = [gt_row[4], gt_row[5]]
            overlap = get_range_overlap(pred_range, gt_range)
            # overlap = time_overlap(pred_range, gt_range)
            if overlap > 0:
                overlaps["sel_table"].append(row.gt_file)
                overlaps["rel_gt_start"].append(gt_row[4])
                overlaps["rel_gt_end"].append(gt_row[5])
                overlaps["rel_pred_start"].append(row.rel_gt_time_start)
                overlaps["rel_pred_end"].append(row.rel_gt_time_end)
    overlap_df = pd.DataFrame(overlaps)
    overlap_df.to_csv("results/true_positive_binary_predictions.csv", index=False)

def count_for_tape(files):
    df = pd.concat([pd.read_csv(f) for f in files], axis=0)
    print(len(df))


if __name__ == '__main__':
    binary = "/home/alex/data/KARAN_ODOM/predict_1"
    f_map = FileMap("results/file_map.json")
    count("/home/alex/data/KARAN_ODOM/predict_1", f_map)
    analyze_positives("/home/alex/data/KARAN_ODOM/analysis/results")
