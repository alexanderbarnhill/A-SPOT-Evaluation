import numpy as np
import os
import datetime

import pandas as pd


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
                datetime.datetime.strptime(f"{start_date}", "%Y%m%d")
                time_idx = date_idx + 1
            except:
                date_idx += 1

        start_date = components[date_idx]
        start_time = components[time_idx]
        self.start_date_time = datetime.datetime.strptime(f"{start_date}T{start_time}", "%Y%m%dT%H%M%S")
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


class SelectionTable:
    def __init__(self, file_path):
        self.file_path = file_path
        self.start_date_time = self.set_start_date_time()
        self.data = self._init_data()


    def set_start_date_time(self):
        basename = os.path.basename(self.file_path).replace(".Table.1.selections.FINAL.txt", "")
        components = basename.split("_")
        date_idx = 1
        for i, c in enumerate(components[0:]):
            try:
                idx = int(c)
                date_idx = i
                break
            except:
                continue
        time_idx = date_idx + 1
        s = f"{components[date_idx]}T{components[time_idx]}"
        return datetime.datetime.strptime(s, "%Y%m%dT%H%M%S")

    def _init_data(self):
        data = pd.read_csv(self.file_path, sep="\t")
        return data


if __name__ == '__main__':
    f = "/media/alex/s1/experiments/ANIMAL-SPOT/warbler/trained_multi_class_model/WARBLER_SEG_AS_MULTI_3CLASS_V4/predictions/2019_22MAY18C12GWAC12GW1T2_predict_output.log"
    g = "/home/alex/data/KARAN_ODOM/FINAL corrected annotations/N9_S00920_20220516_053000.Table.1.selections.FINAL.txt"
    # l = PredictionLog(f)
    h = SelectionTable(g)