import json

class FileMap:
    gt_pred_map = None
    def __init__(self, map_path):
        with open(map_path, "r") as f:
            self.gt_pred_map = json.load(f)