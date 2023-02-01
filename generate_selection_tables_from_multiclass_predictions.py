from datetime import datetime, timedelta
import pandas as pd
import os
import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_dir",
    type=str,
    help="Directory with multiclass log files"
)

parser.add_argument(
    "--log_extension",
    type=str,
    default="output.log",
    help="Suffix on the multiclass log files. Default `'output.log'`"
)

parser.add_argument(
    "--output_dir",
    type=str,
    help="Directory where selection tables will be written"
)

class PredictionLog:
    def __init__(self, file_path, non_noise_threshold=0.9, multiclass=False):
        self.log_file = file_path
        self.audio_file = ""
        self.multiclass = multiclass
        if self.multiclass:
            self.binary_start = None
            self.binary_end = None
        self.gt_start = None
        self.start_date_time = None
        self.data = self._init_data()



    def _get_line_content(self, line):
        components = line.split("|")
        return components[-1]

    def _get_key_value(self, pair):
        components = pair.split("=")
        return components[0], components[1].rstrip()

    def set_start_date(self):
        basename = os.path.basename(self.audio_file).replace(".wav", "")
        components = basename.split("_")
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
        self.start_date_time = datetime.strptime(f"{start_date}T{start_time}", "%Y%m%dT%H%M%S")
        if self.multiclass:
            component = components[0]
            offset_components = component.split("-")
            self.binary_start = float(offset_components[1].replace("ms", "")) / 1000
            self.binary_end = float(offset_components[2].replace("ms", "")) / 1000


    def _group_smooth(self):
        grouped = []

        data = [l for l in self.data if l["class_id"] == 1]
        # data = [l for l in self.data if l["class_id"] != "noise-2" and l["pred"] == 1]
        group = {
            "start_t": data[0]["start_t"],
            "end_t": data[0]["end_t"],
            "class_id": data[0]["class_id"]
        }
        for l in data[1:]:
            if l["class_id"] == group["class_id"] and l["start_t"] < group["end_t"]:
                group["end_t"] = l["end_t"]
            elif l["class_id"] != group["class_id"] and l["start_t"] < group["end_t"]:
                continue
            else:
                grouped.append(group)
                group = {
                    "start_t": l["start_t"],
                    "end_t": l["end_t"],
                    "class_id": l["class_id"]
                }
        return grouped


    def _group_non_smooth(self):
        grouped = []

        start_t = 0.0
        end_t = 0.0
        class_id = None

        for idx, line in enumerate(self.data):
            if class_id is None:
                class_id = line["class_id"]
                end_t = line["end_t"]
                continue
            if line["class_id"] != class_id:
                grouped.append({
                    "start_t": start_t,
                    "end_t": end_t,
                    "class_id": class_id
                })
                start_t = line["start_t"]
                end_t = line["end_t"]
                class_id = line["class_id"]
                continue
            if line["class_id"] == class_id:
                end_t = line["end_t"]

        return grouped

    def group(self, smooth=True):
        if smooth:
            return self._group_smooth()
        return self._group_non_smooth()

    def _init_data(self):
        with open(self.log_file, "r") as f:
            lines = f.readlines()
        data = []
        for idx in range(len(lines)):
            line = lines[idx]
            if idx == 0:
                self.audio_file = self._get_line_content(line).rstrip()
                self.set_start_date()
                continue
            if "|time=" in line:
                content = self._get_line_content(line)
                fields = content.split(", ")
                time = self._get_key_value(fields[0])
                t_start = time[1].split("-")[0]
                t_end = time[1].split("-")[1]
                pred = int(self._get_key_value(fields[1])[1])
                if self.multiclass:
                    pred_class = self._get_key_value(fields[2])[1]
                    prob = self._get_key_value(fields[3])[1]
                    active_entry = {
                        "start_t": float(t_start),
                        "end_t": float(t_end),
                        "class_id": pred_class,
                        "prob": float(prob),
                        "pred": int(pred),
                        "classes": {}
                    }
                    idx += 1
                    line = lines[idx]
                    if "output_layer" in line:
                        idx += 1
                        line = lines[idx]

                    while len(line.rstrip()) > 0:
                        c, p = line.split("=")
                        p = float(p.replace(";", "").rstrip())
                        active_entry["classes"][c] = p
                        idx += 1
                        line = lines[idx]
                else:
                    prob = self._get_key_value(fields[2])[1]
                    active_entry = {
                        "start_t": float(t_start),
                        "end_t": float(t_end),
                        "pred": int(pred),
                        "class_id": int(pred),
                        "prob": prob
                    }
                data.append(active_entry)
        return data


def get_weighted_prediction(data):
    preds = {}
    for p in data:
        for cl in p["classes"]:
            if cl not in preds:
                preds[cl] = 0
            preds[cl] += p["classes"][cl]
    sorted_preds = {k: v for k, v in sorted(preds.items(), key=lambda item: item[1])}
    top_prediction = list(sorted_preds.keys())[-1]
    return top_prediction


class PredictionLogFile:
    def __init__(self, file, multiclass=True):
        print(f"Initializing {file}...")
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
    print(f"Found logs from {len(groups.keys())} different base audio files")
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
    print(f"Writing {name}")
    data = {
        "Selection": [i for i in range(1, len(group) + 1)],
        "View": ["Spectrogram 1" for _ in range(len(group))],
        "Channel": [1 for _ in range(len(group))],
        "Begin Time (s)": [],
        "End Time (s)": [],
        "Prediction": [],
    }
    print(f"Found {len(group)} entries")
    for entry in group:
        prediction = get_weighted_prediction(entry.prediction_log.data)
        data["Begin Time (s)"].append(entry.rel_start_s)
        data["End Time (s)"].append(entry.rel_end_s)
        data["Prediction"].append(prediction)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(output_directory, name), sep="\t", index=False)
    print("-----------------------------")


def make_tables(directory, ext, output_directory):
    logs = [PredictionLogFile(l) for l in get_files(directory, ext)]
    print(f"Found {len(logs)} multiclass log files")
    groups = get_groups(logs)
    for key, value in groups.items():
        make_table(value, output_directory)

ARGS = parser.parse_args()
if __name__ == '__main__':
    # extension = "output.log"
    # folder = "/home/alex/data/KARAN_ODOM/predict_2_EvalSet_20221205/logs"
    # output_dir = "/home/alex/data/KARAN_ODOM/predict_2_EvalSet_20221205"
    input_dir = ARGS.input_dir
    output_dir = ARGS.output_dir
    print(f"Looking for files in {input_dir}")
    print(f"Writing selection tables to {output_dir}")
    make_tables(ARGS.input_dir, ARGS.log_extension, ARGS.output_dir)
