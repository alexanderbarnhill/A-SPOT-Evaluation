import pandas as pd
import json
from file_map import FileMap
from log_models import SelectionTable, PredictionLog
import os

from collections import namedtuple

from utilities import get_weighted_prediction, get_ground_truth

Range = namedtuple('Range', ['start', 'end'])


def analyze_gt(gt_file, file_map: FileMap):
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
    table = SelectionTable(gt_file)
    binary_prediction_logs = file_map.gt_pred_map[gt_file]
    for b_log_f in binary_prediction_logs:
        b_log = PredictionLog(b_log_f)
        multiclass_prediction_logs = file_map.gt_pred_map[gt_file][b_log_f]
        for m_log_f in multiclass_prediction_logs:
            m_log = PredictionLog(m_log_f, multiclass=True)
            m_log.gt_start = table.start_date_time
            prediction = get_weighted_prediction(m_log.data)
            entry, pred_range, entry_range = get_ground_truth(table, b_log, m_log)
            data["ground_truth_file"].append(os.path.basename(gt_file))
            data["binary_prediction_file"].append(os.path.basename(b_log_f))
            data["binary_start"].append(pred_range.start)
            data["binary_end"].append(pred_range.end)
            data["binary_start_rel"].append((pred_range.start - table.start_date_time).total_seconds())
            data["binary_end_rel"].append((pred_range.end - table.start_date_time).total_seconds())

            data["multiclass_prediction_file"].append(os.path.basename(m_log_f))
            data["multiclass_prediction"].append(prediction.lower())
            if entry is not None:
                data["ground_truth"].append(entry.sex)
                data["ground_truth_start_rel"].append((entry_range.start - table.start_date_time).total_seconds())
                data["ground_truth_end_rel"].append((entry_range.end - table.start_date_time).total_seconds())
                data["ground_truth_start"].append(entry_range.start)
                data["ground_truth_end"].append(entry_range.end)
                data["quality"].append(entry.quality)
                data["notes"].append(entry.notes)
                data["song"].append(entry.song)
                data["call_type"].append(entry[17])
                data["likely_sex"].append(entry[18])
            else:
                data["ground_truth"].append('')
                data["ground_truth_start_rel"].append('')
                data["ground_truth_end_rel"].append('')
                data["ground_truth_start"].append('')
                data["ground_truth_end"].append('')
                data["quality"].append('')
                data["notes"].append('')
                data["song"].append('')
                data["call_type"].append('')
                data["likely_sex"].append('')
    return data

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    f_map = FileMap("results/file_map.json")
    gt_pred_dfs = []
    for gt in list(f_map.gt_pred_map.keys()):
        gt_pred_results = pd.DataFrame(analyze_gt(gt, f_map))\
            .sort_values(by=["binary_start"], ascending=True)\
            .reset_index()\
            .drop(columns=["index"])
        gt_pred_dfs.append(gt_pred_results)

    df = pd.concat(gt_pred_dfs, axis=0)
    df.to_csv("results/prediction_results.csv", index=False)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
