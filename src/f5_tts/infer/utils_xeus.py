import json
from pathlib import Path

import joblib
import librosa
import torch
from cached_path import cached_path
from espnet2.tasks.ssl import SSLTask


INFER_DIR = Path(__file__).resolve().parent
xeus_path = str(cached_path("hf://espnet/xeus/model/xeus_checkpoint_old.pth"))
km_path = str(cached_path("hf://SPRINGLab/EZ-VC/kmeans_xeus_500_multilingual.pkl"))
km_model = joblib.load(km_path)
with open(INFER_DIR / "xeus/char_map.json", encoding="utf-8") as f:
    unit_map = json.load(f)

# device = "cuda" if torch.cuda.is_available() else "cpu"

class ApplyKmeans:
    def __init__(self, device):
        self.km_model = km_model
        self.C_np = self.km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.device = device

        self.C = torch.from_numpy(self.C_np).to(device)
        self.Cnorm = torch.from_numpy(self.Cnorm_np).to(device)

    def __call__(self, x):
        x = x.to(self.device)
        dist = (
            x.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(x, self.C)
            + self.Cnorm
        )
        return dist.argmin(dim=1).cpu().numpy()

# apply_kmeans = ApplyKmeans(device)

# Load XEUS model from checkpoint
def load_xeus_model(device):
    xeus_model, _ = SSLTask.build_model_from_file(
        str(INFER_DIR / "xeus/config.yaml"),
        xeus_path,
        device,
    )
    return xeus_model

def cap_consecutive_units(units, max_consecutive):
    if not units:
        return ""

    if max_consecutive is None or max_consecutive <= 0:
        return "".join(str(unit) for unit in units)

    capped_sequence = [units[0]]
    run_length = 1
    for unit in units[1:]:
        if unit == capped_sequence[-1]:
            run_length += 1
            if run_length <= max_consecutive:
                capped_sequence.append(unit)
        else:
            run_length = 1
            capped_sequence.append(unit)
    return "".join(str(unit) for unit in capped_sequence)


def deduplicate_units(units):
    return cap_consecutive_units(units, max_consecutive=1)


def extract_units(
    audio_path,
    xeus_model,
    apply_kmeans,
    device,
    layer_index=14,
    deduplicate=True,
    max_consecutive_units=None,
):

    audio_array, _ = librosa.load(audio_path, sr=16000)

    # Convert to tensor
    wav_lengths = torch.LongTensor([len(audio_array)]).to(device)
    wav_tensor = torch.Tensor([audio_array]).to(device)

    with torch.no_grad():
        # Encode and get hidden states
        outputs = xeus_model.encode(wav_tensor, wav_lengths, use_mask=False, use_final_output=False)

    hidden_states = outputs[0]
    if not -len(hidden_states) <= layer_index < len(hidden_states):
        raise ValueError(f"layer_index must be between 0 and {len(hidden_states) - 1}, got {layer_index}")

    features = hidden_states[layer_index].squeeze()
    # features_tensor = torch.from_numpy(features).cuda()

    units = apply_kmeans(features).tolist()
    # print(units)
    unit_string = "".join([unit_map[str(unit)] for unit in units])
    if max_consecutive_units is not None:
        unit_string = cap_consecutive_units(unit_string, int(max_consecutive_units))
    elif deduplicate:
        unit_string = deduplicate_units(unit_string)
    return unit_string
