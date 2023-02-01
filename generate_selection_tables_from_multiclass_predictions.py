from datetime import datetime, timedelta
import pandas as pd
import os
import glob

from log_models import PredictionLog
from utilities import get_weighted_prediction


class PredictionLogFile:
    def __init__(self, file, multiclass=True):
        base = os.path.basename(file)
        components = base.split("_")
        self.file = file
        self.basename = base
        self.source = "_".join(components[2:]).replace("_predict_output.log", "")
        self.binary_extraction_start_t = float(components[0].split("-")[1].replace("ms", "")) / 1000
        self.binary_extraction_end_t = float(components[0].split("-")[2].replace("ms", "")) / 1000
        self.binary_extraction_idx = int(components[1])

        date_idx = 0
        time_idx = None
        while time_idx is None:
            try:
                start_date = components[date_idx]
                datetime.strptime(f"{start_date}", "%Y%m%d")
                time_idx = date_idx + 1
            except:
                date_idx += 1

        start_date = components[date_idx]
        start_time = components[time_idx]

        self.start_date = components[date_idx]
        self.start_time = components[time_idx]
        self.start_date_time = datetime.strptime(f"{self.start_date}T{self.start_time}", "%Y%m%dT%H%M%S")
        self.rel_start_s = None
        self.rel_end_s = None

        self.prediction_log = PredictionLog(self.file, multiclass=multiclass)


def get_files(directory, ext):
    return glob.glob(f"{directory}/**/*{ext}", recursive=True)


def get_groups(logs):
    groups = {}
    for log in logs:
        if log.source not in groups:
            groups[log.source] = []
        groups[log.source].append(log)
    gathered = gather_groups(groups)
    flattened = {k: flatten_sort_group(v) for k, v in gathered.items()}
    return flattened


def gather_groups(groups):
    gathered_groups = {}
    for key, files in groups.items():
        components = key.split("_")
        tape_day = "_".join(components[0:3])
        if tape_day not in gathered_groups:
            gathered_groups[tape_day] = []
        gathered_groups[tape_day].append(files)
    return gathered_groups

def flatten_sort_group(group):
    flattened = sorted([item for sublist in group for item in sublist], key=lambda x: x.start_date_time)
    start_time = flattened[0].start_date_time
    for item in flattened:
        item.rel_start_s = ((item.start_date_time + timedelta(seconds=item.binary_extraction_start_t)) - start_time).total_seconds()
        item.rel_end_s = ((item.start_date_time + timedelta(seconds=item.binary_extraction_end_t)) - start_time).total_seconds()
    return sorted(flattened, key=lambda x: x.rel_start_s)


def make_table(group, output_directory):
    name = group[0].source + "_prediction_selection_table.txt"
    data = {
        "Selection": [i for i in range(1, len(group) + 1)],
        "View": ["Spectrogram 1" for _ in range(len(group))],
        "Channel": [1 for _ in range(len(group))],
        "Begin Time (s)": [],
        "End Time (s)": [],
        "Prediction": [],
    }

    for entry in group:
        prediction = get_weighted_prediction(entry.prediction_log.data)
        data["Begin Time (s)"].append(entry.rel_start_s)
        data["End Time (s)"].append(entry.rel_end_s)
        data["Prediction"].append(prediction)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_directory, name), sep="\t", index=False)


def make_tables(directory, ext, output_directory):
    logs = [PredictionLogFile(l) for l in get_files(directory, ext)]
    groups = get_groups(logs)
    for key, value in groups.items():
        make_table(value, output_directory)


if __name__ == '__main__':
    extension = "output.log"
    folder = "/home/alex/data/KARAN_ODOM/predict_2_EvalSet_20221205/logs"
    output_dir = "/home/alex/data/KARAN_ODOM/predict_2_EvalSet_20221205"
    make_tables(folder, extension, output_dir)
