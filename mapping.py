import glob
import json
import os
from log_models import PredictionLog

def get_files(directory, ext):
    return glob.glob(f"{directory}/**/*.{ext}", recursive=True)

def get_multiclass_for_binary(multiclass_prediction_logs, bin_file):
    leading = get_leading_component(bin_file, 4)
    wav_file = os.path.basename(bin_file).replace("_predict_output.log", ".wav")
    logs = [m for m in multiclass_prediction_logs if wav_file in m.audio_file]
    files = [m.log_file for m in logs]

    return files

def get_leading_component(file, count=3):
    base_file = os.path.basename(file.replace(".Table.1.selections.FINAL.txt", ""))
    leading_component = "_".join(base_file.split("_")[0:count])
    return leading_component

def make_mapping(gt_folder, p1_folder, p2_folder):
    data = {}

    ground_truth_files = get_files(gt_folder, "txt")
    binary_files = get_files(p1_folder, "log")
    multiclass_files = get_files(p2_folder, "log")
    multiclass_prediction_logs = [PredictionLog(m) for m in multiclass_files]
    total_logs = 0
    bin_logs = 0
    bin_lo = []

    for gt in ground_truth_files:
        leading_component = get_leading_component(gt)

        data[gt] = {}

        binary_predictions = [f for f in binary_files if leading_component in f]
        bin_logs += len(binary_predictions)
        bin_lo += binary_predictions
        for b in binary_predictions:
            m = get_multiclass_for_binary(multiclass_prediction_logs, b)
            total_logs += len(m)
            data[gt][b] = m

    with open("results/file_map.json", "w") as f:
        json.dump(data, f, indent=4)
    for f in binary_files:
        if f not in bin_lo:
            print(f)
    print(f"Binary Logs: {bin_logs}")
    print(f"Multiclass Logs: {total_logs}")



if __name__ == '__main__':
    ground_truth = "/home/alex/data/KARAN_ODOM/FINAL corrected annotations"
    binary = "/home/alex/data/KARAN_ODOM/predict_1"
    multiclass = "/home/alex/data/KARAN_ODOM/predict_2_EvalSet_20221205"

    make_mapping(ground_truth, binary, multiclass)
