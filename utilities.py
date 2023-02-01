from datetime import timedelta
from collections import namedtuple
Range = namedtuple('Range', ['start', 'end'])


def overlap_percentage(xlist, ylist):
    min1 = min(xlist)
    max1 = max(xlist)
    min2 = min(ylist)
    max2 = max(ylist)

    overlap = max(0, min(max1, max2) - max(min1, min2))
    length = max1-min1 + max2-min2
    lengthx = max1-min1
    lengthy = max2-min2

    return 2 * overlap/length, overlap/lengthx, overlap/lengthy


def time_overlap(xlist, ylist):
    a, _, _ = overlap_percentage(xlist, ylist)
    return a


def get_range_overlap(r1, r2):
    latest_start = max(r1.start, r2.start)
    earliest_end = min(r1.end, r2.end)
    delta = (earliest_end - latest_start).days + 1
    overlap = max(0, delta)
    return overlap


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


def get_ground_truth(table, binary_log, multiclass_log):
    pred_offset = (binary_log.start_date_time - table.start_date_time).total_seconds()
    prediction_start = multiclass_log.start_date_time + timedelta(seconds=multiclass_log.binary_start)
    prediction_end = multiclass_log.start_date_time + timedelta(seconds=multiclass_log.binary_end)
    r1 = Range(start=prediction_start, end=prediction_end)
    for entry in table.data.itertuples():
        entry_start = table.start_date_time + timedelta(seconds=entry[4])
        entry_end = table.start_date_time + timedelta(seconds=entry[5])
        r2 = Range(start=entry_start, end=entry_end)
        overlap = get_range_overlap(r1, r2)
        if overlap > 0:
            return entry, r1, r2
    return None, r1, None
