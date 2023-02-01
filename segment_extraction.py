import os
import argparse
import soundfile as sf
import torch
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    dest="input",
    help="File or folder with files to be extracted",
)

parser.add_argument(
    "-a",
    "--audio_output",
    dest="audio_output",
    help="Where the extracted segments will be stored",
)

parser.add_argument(
    "-s",
    "--spectrogram_output",
    dest="spectrogram_output",
    help="Where the spectrograms of the extracted segments will be stored. If empty, no spectrograms will be created",
)

parser.add_argument(
    "--smooth",
    dest="smooth",
    action="store_true",
    default=True,
    help="If set, neighboring frames which are positive will be grouped to create longer segments. Default true",
)

parser.add_argument(
    "--threshold",
    dest="threshold",
    type=float,
    help="A threshold for marking if a segment should be extracted. If not set, the threshold in the prediction file will be used",
)



class Spectrogram(object):
    """Converts a given audio to a spectrogram.

    Args:
        n_fft (int): FFT window size.
        hop_length (int): FFT hop size.
        power (bool): Whether to return a power spectrogram. Defaults to `True`.
    """

    def __init__(self, n_fft, hop_length, center=True, return_complex=False):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.center = center
        self.window = torch.hann_window(self.n_fft)
        self.return_complex = return_complex

    def __call__(self, y):
        if y.dim() != 2:
            raise ValueError(
                "Spectrogram expects a 2 dimensional signal of size (c, n), "
                "but got size: {}.".format(y.size())
            )
        S = torch.stft(
            input=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            center=self.center,
            onesided=True
        ).transpose(1, 2)
        S /= self.window.pow(2).sum().sqrt()
        if not self.return_complex:
            return S.pow(2).sum(-1)  # get power of "complex" tensor (c, l, n_fft)
        else:
            return S


class PredictionLog:
    def __init__(self, file_path, non_noise_threshold=None, binary=True):
        self.log_file = file_path
        self.audio_file = ""
        self.binary = binary
        self.used_threshold = non_noise_threshold
        self.data = self._init_data()

    def is_positive(self, probability, prediction):
        if self.used_threshold is not None:
            return probability > self.used_threshold
        return prediction

    def _get_line_content(self, line):
        components = line.split("|")
        return components[-1]

    def _get_key_value(self, pair):
        components = pair.split("=")
        return components[0], components[1].rstrip()

    def _group_smooth(self):
        grouped = []

        data = [l for l in self.data if self.is_positive(float(l["prob"]), int(l["pred"]))]
        if len(data) == 0:
            return grouped

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
                self.audio_file = self._get_line_content(line)
                continue
            if "|time=" in line:
                content = self._get_line_content(line)
                fields = content.split(", ")
                time = self._get_key_value(fields[0])
                t_start = time[1].split("-")[0]
                t_end = time[1].split("-")[1]
                pred = int(self._get_key_value(fields[1])[1])
                if self.binary:
                    prob = float(self._get_key_value(fields[2])[1])
                    pred_class = "target" if self.is_positive(prob, pred) else "noise"
                else:
                    pred_class = self._get_key_value(fields[2])[1]
                    prob = float(self._get_key_value(fields[3])[1])

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

                if not self.binary:
                    while len(line.rstrip()) > 0:
                        c, p = line.split("=")
                        p = float(p.replace(";", "").rstrip())
                        active_entry["classes"][c] = p
                        idx += 1
                        line = lines[idx]
                data.append(active_entry)
        return data

    def load_spectrogram(self, audio_data, title: str, output: str):
        spectrogram = Spectrogram(hop_length=128, n_fft=1024)(torch.from_numpy(audio_data).unsqueeze(0))[0].T
        figure, ax = plt.subplots(dpi=300)
        plt.title(title)
        os.makedirs(output, exist_ok=True)
        plt.imshow(torch.log(spectrogram), origin="lower", interpolation=None)
        plt.savefig(os.path.join(output, f"{title.replace(' ', '')}.png"), bbox_inches="tight")
        plt.close(figure)


    def get_segment_name(self, start_s, end_s, idx):
        basename = os.path.basename(self.audio_file.strip())
        basename = f"target-{int(start_s * 1000)}ms-{int(end_s * 1000)}ms_{idx}_{basename}"
        return basename


    def segment(self, audio_output, img_output=None, smooth_predictions=True):
        grouped_events = self.group(smooth_predictions)
        source_audio, sr = sf.read(self.audio_file.strip())
        os.makedirs(audio_output, exist_ok=True)
        if img_output is not None:
            os.makedirs(img_output, exist_ok=True)

        for idx, event in enumerate(grouped_events):
            audio_slice = source_audio[int(event['start_t'] * sr): int(event['end_t'] * sr)]
            name = self.get_segment_name(event['start_t'], event['end_t'], idx)
            print(name)
            if img_output is not None:
                self.load_spectrogram(audio_slice, name, img_output)
            sf.write(os.path.join(audio_output, name), audio_slice, sr)


def run_segment(f, args):
    pl = PredictionLog(f, args.threshold)
    pl.segment(args.audio_output, args.spectrogram_output, args.smooth)


if __name__ == '__main__':
    ARGS = parser.parse_args()
    if os.path.isdir(ARGS.input):
        files = [os.path.join(ARGS.input, f) for f in os.listdir(ARGS.input) if f.endswith("predict_output.log")]
    else:
        files = [ARGS.input]
    for file in files:
        run_segment(file, ARGS)
