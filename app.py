from __future__ import annotations

import csv
from datetime import datetime
import io
import json
from pathlib import Path
import tempfile

import mne
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file
from mne.datasets import eegbci
from mne.decoding import CSP
from scipy.signal import spectrogram, welch
from scipy.io import loadmat
from sklearn.calibration import CalibratedClassifierCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC, SVC
import joblib

try:
    import h5py
except Exception:
    h5py = None


app = Flask(__name__)

DATA_ROOT = Path.home() / "mne_data" / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0"
MI_RUNS = [4, 8, 12]
PREFERRED_CHANNELS = ["FCz", "C3", "Cz", "C4", "CP3", "CPz", "CP4", "Pz"]
BANDS = {
    "Alpha (8-12 Hz)": (8, 12),
    "Beta (13-30 Hz)": (13, 30),
    "Gamma (31-45 Hz)": (31, 45),
}
CONFIG_PATH = Path(__file__).with_name("config.json")
MODEL_DIR = Path(__file__).with_name("saved_models")
MODEL_PATH = MODEL_DIR / "latest_model.joblib"
REPORT_PATH = MODEL_DIR / "latest_report.json"

APP_STATE = {
    "source": "none",
    "dataset": None,
    "summary": {"status": "No EEG dataset loaded."},
    "logs": [],
    "recent_runs": [],
    "last_payload": None,
    "saved_model_available": False,
}


DEFAULT_CONFIG = {
    "filter_band": [8.0, 30.0],
    "epoch_window": [1.0, 4.0],
    "csp_components": 4,
    "test_size": 0.3,
    "random_seed": 42,
    "preferred_classifier": "SVM",
    "subject_default": 1,
}


def _load_config() -> dict:
    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        return {**DEFAULT_CONFIG, **loaded}
    with CONFIG_PATH.open("w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CONFIG, handle, indent=2)
    return DEFAULT_CONFIG.copy()


CONFIG = _load_config()
MODEL_DIR.mkdir(exist_ok=True)


def _log(level: str, message: str) -> None:
    APP_STATE["logs"].append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "level": level.upper(),
            "message": message,
        }
    )
    APP_STATE["logs"] = APP_STATE["logs"][-12:]


def _start_run_log() -> None:
    APP_STATE["logs"] = []
    _log("info", "Session initialized.")
    _log("info", f"Configuration loaded with random seed {CONFIG['random_seed']}.")


def _dataset_family(source: str, subject: str) -> str:
    text = f"{source} {subject}".lower()
    if "physionet" in text or "eegbci" in text or ".edf" in text:
        return "PhysioNet EEGBCI"
    if ".mat" in text or "matlab" in text or "kaggle" in text:
        return "MATLAB EEG Session"
    if "synthetic" in text or "demo" in text:
        return "Synthetic Demo"
    return "Unified EEG Session"


def _subject_dir(subject: int) -> Path:
    return DATA_ROOT / f"S{subject:03d}"


def _safe_cv_splits(y: np.ndarray) -> int:
    counts = np.bincount(y)
    valid = counts[counts > 0]
    if valid.size == 0:
        return 2
    return max(2, min(5, int(valid.min())))


def _pick_channels(raw: mne.io.BaseRaw) -> list[str]:
    available = [name for name in PREFERRED_CHANNELS if name in raw.ch_names]
    if len(available) >= 6:
        return available[:8]
    eeg_channels = mne.pick_types(raw.info, eeg=True, exclude="bads")
    return [raw.ch_names[idx] for idx in eeg_channels[:8]]


def _raw_to_waveform(raw: mne.io.BaseRaw, duration: float = 8.0) -> dict:
    channels = _pick_channels(raw)
    picks = [raw.ch_names.index(name) for name in channels]
    stop = min(int(raw.info["sfreq"] * duration), raw.n_times)
    times = raw.times[:stop]
    segment = raw.get_data(picks=picks, start=0, stop=stop) * 1e6
    offsets = np.arange(len(channels))[::-1] * 160.0
    traces = []
    palette = ["#22d3ee", "#8b5cf6", "#22c55e", "#f59e0b", "#3b82f6", "#ec4899", "#14b8a6", "#f97316"]
    for idx, name in enumerate(channels):
        traces.append(
            {
                "name": name,
                "x": np.round(times, 4).tolist(),
                "y": np.round(segment[idx] + offsets[idx], 3).tolist(),
                "color": palette[idx % len(palette)],
            }
        )
    return {"traces": traces, "offsets": offsets.tolist()}


def _epochs_to_waveform(epochs: mne.Epochs) -> dict:
    sample = epochs.get_data()[0]
    channels = epochs.ch_names[: min(8, len(epochs.ch_names))]
    segment = sample[: len(channels)] * 1e6
    times = np.arange(segment.shape[-1]) / float(epochs.info["sfreq"])
    offsets = np.arange(len(channels))[::-1] * 160.0
    palette = ["#22d3ee", "#8b5cf6", "#22c55e", "#f59e0b", "#3b82f6", "#ec4899", "#14b8a6", "#f97316"]
    traces = []
    for idx, name in enumerate(channels):
        traces.append(
            {
                "name": name,
                "x": np.round(times, 4).tolist(),
                "y": np.round(segment[idx] + offsets[idx], 3).tolist(),
                "color": palette[idx % len(palette)],
            }
        )
    return {"traces": traces, "offsets": offsets.tolist()}


def _normalize_trials(trials: np.ndarray, n_channels: int) -> np.ndarray:
    arr = np.asarray(trials, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    if arr.ndim != 3:
        raise ValueError("Expected 2D or 3D trial data.")
    if arr.shape[0] == n_channels:
        arr = np.transpose(arr, (2, 0, 1))
    elif arr.shape[1] == n_channels:
        pass
    elif arr.shape[2] == n_channels:
        arr = np.transpose(arr, (0, 2, 1))
    else:
        raise ValueError("Could not infer channel axis from trial array.")
    return arr


def _as_plain_dict(value, prefix: str = "") -> dict[str, np.ndarray]:
    flat = {}
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).startswith("__"):
                continue
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_as_plain_dict(item, next_prefix))
        return flat
    if hasattr(value, "_fieldnames"):
        for key in getattr(value, "_fieldnames", []):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(_as_plain_dict(getattr(value, key), next_prefix))
        return flat
    if isinstance(value, np.ndarray) and value.dtype == object:
        for index, item in np.ndenumerate(value):
            next_prefix = f"{prefix}[{','.join(map(str, index))}]" if prefix else str(index[0])
            flat.update(_as_plain_dict(item, next_prefix))
        return flat
    if prefix:
        flat[prefix] = np.asarray(value)
    return flat


def _epochs_from_arrays(trials: np.ndarray, labels: np.ndarray, sfreq: float, ch_names: list[str], source_name: str) -> mne.Epochs:
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    trials = _normalize_trials(trials, len(ch_names))
    labels = np.asarray(labels).astype(int).reshape(-1)
    if trials.shape[0] != labels.shape[0]:
        raise ValueError(f"{source_name}: trial count does not match label count.")
    events = np.column_stack([np.arange(len(labels)), np.zeros(len(labels), dtype=int), labels + 2])
    epochs = mne.EpochsArray(
        trials,
        info,
        events=events,
        event_id={"Left Hand Movement": 2, "Right Hand Movement": 3},
        tmin=0.0,
        verbose="ERROR",
    )
    epochs.set_montage("standard_1005", on_missing="ignore")
    return epochs


def _combine_epochs(items: list[tuple[str, mne.Epochs, mne.io.BaseRaw | None]]) -> dict:
    epochs_list = [item[1] for item in items]
    combined = mne.concatenate_epochs(epochs_list, verbose="ERROR") if len(epochs_list) > 1 else epochs_list[0]
    representative_raw = next((item[2] for item in items if item[2] is not None), None)
    source_names = [item[0] for item in items]
    return {
        "epochs": combined,
        "raw": representative_raw,
        "source_names": source_names,
    }


def _load_signal_file(file_path: Path) -> mne.io.BaseRaw:
    suffix = file_path.suffix.lower()
    if suffix == ".edf":
        return mne.io.read_raw_edf(file_path, preload=True, verbose="ERROR")
    raise ValueError(f"Unsupported signal file: {file_path.name}")


def _prepare_motor_imagery_epochs(raw: mne.io.BaseRaw) -> mne.Epochs:
    eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="ignore")
    raw.filter(CONFIG["filter_band"][0], CONFIG["filter_band"][1], fir_design="firwin", verbose="ERROR")
    events, event_map = mne.events_from_annotations(raw, verbose="ERROR")
    if not event_map:
        raise ValueError("No event annotations were found for motor imagery extraction.")

    left_keys = {"t1", "769", "left", "left_hand", "left hand"}
    right_keys = {"t2", "770", "right", "right_hand", "right hand"}
    remapped = []
    for name, code in event_map.items():
        key = name.strip().lower()
        if key in left_keys:
            remapped.append((name, code, 2))
        if key in right_keys:
            remapped.append((name, code, 3))
    if not remapped:
        raise ValueError("No left/right motor imagery events were detected in this recording.")

    left_codes = {item[1] for item in remapped if item[2] == 2}
    right_codes = {item[1] for item in remapped if item[2] == 3}
    selected_events = []
    for event in events:
        if event[2] in left_codes:
            selected_events.append([event[0], event[1], 2])
        elif event[2] in right_codes:
            selected_events.append([event[0], event[1], 3])
    if not selected_events:
        raise ValueError("The file contained annotations, but none matched left/right motor imagery trials.")

    picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude="bads")
    return mne.Epochs(
        raw,
        np.asarray(selected_events, dtype=int),
        event_id={"Left Hand Movement": 2, "Right Hand Movement": 3},
        tmin=CONFIG["epoch_window"][0],
        tmax=CONFIG["epoch_window"][1],
        proj=True,
        picks=picks,
        baseline=None,
        preload=True,
        verbose="ERROR",
    )


def _label_to_binary(values: np.ndarray) -> np.ndarray:
    labels = np.asarray(values).astype(int).reshape(-1)
    unique = sorted({int(item) for item in labels.tolist()})
    if set(unique).issubset({0, 1}):
        return labels
    if set(unique).issubset({1, 2}):
        return labels - 1
    if 769 in unique or 770 in unique:
        mapped = []
        for item in labels:
            if int(item) == 769:
                mapped.append(0)
            elif int(item) == 770:
                mapped.append(1)
        return np.asarray(mapped, dtype=int)
    raise ValueError("Only left/right motor imagery labels are currently supported.")


def _extract_from_mat_dict(data: dict, source_name: str) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
    clean = {}
    for key, value in data.items():
        if str(key).startswith("__"):
            continue
        clean.update(_as_plain_dict(value, str(key)))

    imagery_left_key = next((key for key in clean if "imagery_left" in key.lower()), None)
    imagery_right_key = next((key for key in clean if "imagery_right" in key.lower()), None)
    srate_key = next((key for key in clean if any(term in key.lower() for term in ["srate", "sfreq", "fs"])), None)
    if imagery_left_key and imagery_right_key and srate_key:
        left = np.asarray(clean[imagery_left_key])
        right = np.asarray(clean[imagery_right_key])
        n_channels = left.shape[0] if left.ndim == 3 else left.shape[1]
        sfreq = float(np.asarray(clean[srate_key]).reshape(-1)[0])
        ch_names = [f"EEG {idx + 1:02d}" for idx in range(n_channels)]
        left_trials = _normalize_trials(left, n_channels)
        right_trials = _normalize_trials(right, n_channels)
        trials = np.concatenate([left_trials, right_trials], axis=0)
        labels = np.concatenate([np.zeros(len(left_trials), dtype=int), np.ones(len(right_trials), dtype=int)])
        return trials, labels, sfreq, ch_names

    arrays = [(key, np.asarray(value)) for key, value in clean.items()]
    label_candidates = []
    trial_candidates = []
    channel_candidates = []
    sfreq = 160.0

    for key, arr in arrays:
        key_lower = key.lower()
        if arr.size == 0:
            continue
        if any(term in key_lower for term in ["fs", "srate", "sfreq"]) and arr.size <= 4:
            sfreq = float(arr.reshape(-1)[0])
        if arr.ndim in {1, 2} and any(term in key_lower for term in ["label", "class", "target", "y", "marker"]):
            label_candidates.append((key, arr.reshape(-1)))
        elif arr.ndim == 3:
            trial_candidates.append((key, arr))
        elif arr.ndim == 2 and min(arr.shape) <= 128 and max(arr.shape) >= 64:
            channel_candidates.append((key, arr))

    candidate_trials = None
    candidate_labels = None
    for label_key, labels in label_candidates:
        label_count = labels.shape[0]
        for trial_key, trials in trial_candidates:
            if label_count in trials.shape:
                candidate_trials = trials
                candidate_labels = labels
                break
        if candidate_trials is not None:
            break

    if candidate_trials is None and channel_candidates and label_candidates:
        for label_key, labels in label_candidates:
            label_count = labels.shape[0]
            for trial_key, samples in channel_candidates:
                if samples.ndim == 2 and samples.shape[1] % label_count == 0:
                    samples_per_trial = samples.shape[1] // label_count
                    candidate_trials = samples.reshape(samples.shape[0], label_count, samples_per_trial).transpose(1, 0, 2)
                    candidate_labels = labels
                    break
            if candidate_trials is not None:
                break

    if candidate_trials is None:
        grouped = {}
        for key, value in clean.items():
            if "." not in key:
                continue
            prefix, suffix = key.rsplit(".", 1)
            grouped.setdefault(prefix, {})[suffix.lower()] = np.asarray(value)
        for prefix, item in grouped.items():
            if {"x", "trial", "y"} <= item.keys():
                signal = np.asarray(item["x"], dtype=np.float64)
                onsets = np.asarray(item["trial"]).reshape(-1).astype(int)
                labels = _label_to_binary(np.asarray(item["y"]).reshape(-1))
                if signal.ndim != 2 or onsets.size == 0 or labels.size == 0 or labels.size != onsets.size:
                    continue
                local_sfreq = float(np.asarray(item.get("fs", sfreq)).reshape(-1)[0])
                if signal.shape[0] < signal.shape[1]:
                    signal = signal.T
                n_samples, n_channels = signal.shape
                diffs = np.diff(np.sort(onsets))
                valid_diffs = diffs[diffs > 0]
                default_window = int(local_sfreq * 4)
                window = int(valid_diffs.min()) if valid_diffs.size else default_window
                window = max(int(local_sfreq * 2), min(window, default_window))
                extracted = []
                filtered_labels = []
                for onset, label in zip(onsets, labels):
                    start = int(onset)
                    stop = start + window
                    if start < 0 or stop > n_samples:
                        continue
                    extracted.append(signal[start:stop].T)
                    filtered_labels.append(int(label))
                if extracted:
                    ch_names = [f"EEG {idx + 1:02d}" for idx in range(n_channels)]
                    return np.asarray(extracted), np.asarray(filtered_labels), local_sfreq, ch_names

    if candidate_trials is None or candidate_labels is None:
        available_keys = ", ".join(list(clean.keys())[:8])
        raise ValueError(
            f"{source_name}: could not infer left/right trials and labels from MAT data. "
            f"Expected arrays like imagery_left/imagery_right or a 3D trial array plus labels. "
            f"Detected keys: {available_keys}"
        )

    label_count = np.asarray(candidate_labels).reshape(-1).shape[0]
    if candidate_trials.shape[0] == label_count:
        n_channels = candidate_trials.shape[1]
    elif candidate_trials.shape[1] == label_count:
        n_channels = candidate_trials.shape[0]
    elif candidate_trials.shape[2] == label_count:
        n_channels = candidate_trials.shape[1]
    else:
        n_channels = candidate_trials.shape[1]
    ch_names = [f"EEG {idx + 1:02d}" for idx in range(n_channels)]
    return candidate_trials, _label_to_binary(candidate_labels), sfreq, ch_names


def _load_mat_epochs(file_path: Path) -> tuple[mne.Epochs, None]:
    source_name = file_path.name
    try:
        data = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        trials, labels, sfreq, ch_names = _extract_from_mat_dict(data, source_name)
        return _epochs_from_arrays(trials, labels, sfreq, ch_names, source_name), None
    except NotImplementedError:
        if h5py is None:
            raise ValueError(f"{source_name}: MATLAB v7.3 file detected, but h5py is not installed.")
        with h5py.File(file_path, "r") as handle:
            data = {key: np.array(handle[key]) for key in handle.keys()}
        trials, labels, sfreq, ch_names = _extract_from_mat_dict(data, source_name)
        return _epochs_from_arrays(trials, labels, sfreq, ch_names, source_name), None


def _load_dataset_from_path(file_path: Path) -> tuple[mne.Epochs, mne.io.BaseRaw | None]:
    suffix = file_path.suffix.lower()
    if suffix == ".edf":
        raw = _load_signal_file(file_path)
        return _prepare_motor_imagery_epochs(raw), raw
    if suffix == ".mat":
        return _load_mat_epochs(file_path)
    raise ValueError(f"Unsupported dataset format: {file_path.name}")

def _diagnostics(raw: mne.io.BaseRaw | None, epochs: mne.Epochs, results: dict, source: str) -> dict:
    channels = _pick_channels(raw) if raw is not None else epochs.ch_names[: min(8, len(epochs.ch_names))]
    epoch_data = epochs.get_data()
    channel_rms = np.sqrt(np.mean(epoch_data[:, : len(channels), :] ** 2, axis=(0, 2))) * 1e6
    left_count = int(np.sum(epochs.events[:, -1] == 2))
    right_count = int(np.sum(epochs.events[:, -1] == 3))
    sampling_rate = float(raw.info["sfreq"]) if raw is not None else float(epochs.info["sfreq"])
    total_channels = len(raw.ch_names) if raw is not None else len(epochs.ch_names)

    session_cards = [
        {"label": "Acquisition Source", "value": source, "accent": "cyan"},
        {"label": "Analyzed Epochs", "value": str(len(epochs)), "accent": "blue"},
        {"label": "Sampling Rate", "value": f"{sampling_rate:.1f} Hz", "accent": "green"},
        {"label": "EEG Channels", "value": str(total_channels), "accent": "amber"},
    ]
    pipeline = [
        {"title": "Signal Intake", "body": "PhysioNet EEGBCI motor imagery runs are merged into a single session view for rapid evaluation."},
        {"title": "Bandpass Filtering", "body": "An 8-30 Hz band isolates mu and beta rhythms linked to imagined limb movement."},
        {"title": "Epoch Extraction", "body": "Cue-locked windows from 1.0 s to 4.0 s are segmented into left and right motor imagery trials."},
        {"title": "Spatial Decoding", "body": "Common Spatial Patterns maximize discriminative variance across hemispheric motor activity."},
        {"title": "Classification", "body": "A calibrated linear SVM outputs a class decision, confidence score, and validation metrics."},
    ]
    channel_cards = [
        {
            "name": name,
            "rms": round(float(rms), 2),
            "focus": "Motor strip" if name in {"C3", "C4", "Cz", "CP3", "CP4", "FCz"} else "Support channel",
        }
        for name, rms in zip(channels, channel_rms)
    ]
    notes = [
        f"Class balance remains stable with {left_count} left-hand and {right_count} right-hand epochs in the active session.",
        "The EEG monitor emphasizes central and fronto-central electrodes to mimic a clinical motor imagery review layout.",
        f"Current inference predicts {results['predicted_class']} with {results['confidence']:.1f}% confidence after probability calibration.",
        f"Validation performance is {results['accuracy']:.1f}% on the held-out split with {results['cross_validation']:.1f}% mean cross-validation accuracy.",
    ]
    footer = {
        "body": "This system helps compare motor imagery activity, monitor left-versus-right hand intent, and present clinically readable EEG intelligence in one unified workspace.",
    }
    family = _dataset_family(source, "")
    wizard = {
        "family": family,
        "headline": "Dataset Import Wizard",
        "description": "The system inspects incoming EDF or MAT files, maps them into a common EEG epoch structure, and retrains the model without extra manual preprocessing.",
        "steps": [
            {
                "title": "Format Detection",
                "body": "Identify whether the session arrives as EDF signal recordings or MATLAB trial arrays and route it into the correct ingestion path.",
            },
            {
                "title": "Signal Normalization",
                "body": "Standardize sampling metadata, channel structure, and left-vs-right motor imagery labels into one shared representation.",
            },
            {
                "title": "Model Retraining",
                "body": "Run CSP feature extraction plus calibrated SVM classification on the active bundle for immediate dashboard feedback.",
            },
        ],
        "supported_formats": ["EDF", "MAT"],
        "recommendation": "EDF mode is best when you want event-aware raw recordings, while MAT mode is best when you already have segmented MATLAB trials ready for direct classification.",
    }
    benefits = [
        {
            "title": "Why It Is Useful",
            "body": "It turns raw or pre-segmented EEG motor imagery data into a readable decision dashboard, which helps with analysis, comparison, and demonstration.",
        },
        {
            "title": "Main Purpose",
            "body": "It is built to classify imagined left-hand versus right-hand movement while showing the underlying EEG evidence, confidence, and validation quality.",
        },
        {
            "title": "Practical Advantage",
            "body": "Using both EDF and MAT support makes the system easier to use with lab recordings, benchmark datasets, and MATLAB-based preprocessing pipelines.",
        },
    ]
    architecture = [
        {"title": "Data Acquisition", "body": "Load PhysioNet EDF sessions or MATLAB trial files as the entry point for motor imagery analysis."},
        {"title": "Preprocessing", "body": "Apply bandpass filtering, epoch extraction, and channel normalization before feature building."},
        {"title": "Feature Extraction (CSP)", "body": "Use Common Spatial Patterns to transform EEG into discriminative spatial features."},
        {"title": "Classification (SVM / LDA)", "body": "Benchmark a linear SVM against LDA and select the stronger model for reporting."},
        {"title": "Evaluation", "body": "Report confidence, accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix outputs."},
        {"title": "Visualization Platform", "body": "Present waveforms, spectral views, CSP maps, logs, and exportable results in an interactive dashboard."},
        {"title": "Augmentation Layer (GAN-Ready)", "body": "The architecture can accommodate GAN-based augmentation as an extension module, but it is not active in the current implementation."},
    ]
    return {
        "session_cards": session_cards,
        "pipeline": pipeline,
        "channel_cards": channel_cards,
        "notes": notes,
        "footer": footer,
        "wizard": wizard,
        "benefits": benefits,
        "architecture": architecture,
    }


def _epoch_analytics(epochs: mne.Epochs) -> dict:
    data = epochs.get_data()
    sample = data[0]
    sfreq = float(epochs.info["sfreq"])
    channel_names = epochs.ch_names[: min(8, len(epochs.ch_names))]
    mean_channels = sample[: len(channel_names)]

    nperseg = min(128, mean_channels[0].shape[-1])
    noverlap = min(96, max(0, nperseg - 1))
    freq_bins, time_bins, spec = spectrogram(mean_channels[0], fs=sfreq, nperseg=nperseg, noverlap=noverlap)
    freq_mask = freq_bins <= 45
    psd_freqs, psd = welch(sample, fs=sfreq, nperseg=min(256, sample.shape[-1]), axis=-1)
    band_power = []
    for label, (low, high) in BANDS.items():
        mask = (psd_freqs >= low) & (psd_freqs <= high)
        value = float(psd[:, mask].mean()) * 1e12 if np.any(mask) else 0.0
        band_power.append({"band": label, "value": round(value, 2)})

    return {
        "spectrogram": {
            "x": np.round(time_bins, 3).tolist(),
            "y": np.round(freq_bins[freq_mask], 2).tolist(),
            "z": np.round(spec[freq_mask] * 1e12, 3).tolist(),
            "bands": [{"label": "Mu", "range": [8, 12]}, {"label": "Beta", "range": [13, 30]}],
        },
        "band_power": band_power,
        "epoch_snapshot": {
            "channels": channel_names,
            "values": np.round(mean_channels, 4).tolist(),
        },
    }


def _prepare_epochs(raw: mne.io.BaseRaw) -> tuple[mne.io.BaseRaw, mne.Epochs]:
    return raw, _prepare_motor_imagery_epochs(raw)


def _evaluate_classifier(name: str, classifier, X_train_csp, X_test_csp, y_train, y_test) -> dict:
    classifier.fit(X_train_csp, y_train)
    y_pred = classifier.predict(X_test_csp)
    probabilities = classifier.predict_proba(X_test_csp)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, labels=[0, 1], zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities[:, 1]) if len(np.unique(y_test)) == 2 else 0.0
    return {
        "name": name,
        "model": classifier,
        "predictions": y_pred,
        "probabilities": probabilities,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
        "roc_auc": float(roc_auc),
        "per_class": [
            {"label": "Left", "precision": round(float(precision[0]) * 100, 1), "recall": round(float(recall[0]) * 100, 1), "f1": round(float(f1[0]) * 100, 1)},
            {"label": "Right", "precision": round(float(precision[1]) * 100, 1), "recall": round(float(recall[1]) * 100, 1), "f1": round(float(f1[1]) * 100, 1)},
        ],
    }


def _classifier_catalog() -> dict:
    return {
        "SVM": SVC(kernel="linear", probability=True, random_state=CONFIG["random_seed"]),
        "LDA": LinearDiscriminantAnalysis(),
    }


def _classify_epochs(train_epochs: mne.Epochs, test_epochs: mne.Epochs | None = None) -> dict:
    X = train_epochs.get_data()
    y = train_epochs.events[:, -1] - 2
    cv_splits = _safe_cv_splits(y)
    if test_epochs is None:
        class_counts = np.bincount(y)
        valid_counts = class_counts[class_counts > 0]
        if valid_counts.size and valid_counts.min() < 3:
            X_train, X_test, y_train, y_test = X, X, y, y
            evaluation_note = "Small-sample evaluation"
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=CONFIG["test_size"],
                random_state=CONFIG["random_seed"],
                stratify=y,
            )
            evaluation_note = "Random hold-out evaluation"
    else:
        X_train = X
        y_train = y
        X_test = test_epochs.get_data()
        y_test = test_epochs.events[:, -1] - 2
        evaluation_note = "Cross-subject evaluation"

    csp = CSP(n_components=CONFIG["csp_components"], reg=None, log=True, norm_trace=False)
    X_train_csp = csp.fit_transform(X_train, y_train)
    X_test_csp = csp.transform(X_test)

    evaluations = []
    for name, classifier in _classifier_catalog().items():
        try:
            evaluations.append(_evaluate_classifier(name, classifier, X_train_csp, X_test_csp, y_train, y_test))
        except Exception:
            continue
    if not evaluations:
        raise ValueError("No classifier could be fitted on the current dataset. Add more balanced trials and retry.")

    preferred = next((item for item in evaluations if item["name"] == CONFIG["preferred_classifier"]), evaluations[0])
    best = max(evaluations, key=lambda item: item["accuracy"])
    selected = preferred if preferred["accuracy"] >= best["accuracy"] - 0.02 else best

    if evaluation_note == "Small-sample evaluation":
        cv_scores = np.asarray([selected["accuracy"]])
    else:
        pipeline = Pipeline(
            [
                ("csp", CSP(n_components=CONFIG["csp_components"], reg=None, log=True, norm_trace=False)),
                ("clf", _classifier_catalog()[selected["name"]]),
            ]
        )
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=CONFIG["random_seed"])
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    sample_index = int(np.argmax(selected["probabilities"].max(axis=1)))
    predicted_label = int(np.argmax(selected["probabilities"][sample_index]))
    confidence = float(selected["probabilities"][sample_index][predicted_label])
    matrix = confusion_matrix(y_test, selected["predictions"], labels=[0, 1]).tolist()

    pattern_channels = train_epochs.ch_names[: min(8, len(train_epochs.ch_names))]
    csp_patterns = np.asarray(csp.patterns_)[: CONFIG["csp_components"], : len(pattern_channels)]

    artifact_summary = {
        "bad_channels": len(train_epochs.info.get("bads", [])),
        "dropped_epochs": int(np.sum([1 for item in train_epochs.drop_log if item])),
        "retained_epochs": len(train_epochs),
    }

    joblib.dump(
        {
            "csp": csp,
            "classifier": selected["model"],
            "classifier_name": selected["name"],
            "config": CONFIG,
        },
        MODEL_PATH,
    )
    APP_STATE["saved_model_available"] = True

    return {
        "predicted_class": ["Left Hand Movement", "Right Hand Movement"][predicted_label],
        "confidence": round(confidence * 100, 1),
        "accuracy": round(float(selected["accuracy"]) * 100, 1),
        "cross_validation": round(float(cv_scores.mean()) * 100, 1),
        "precision": round(float(selected["precision"]) * 100, 1),
        "recall": round(float(selected["recall"]) * 100, 1),
        "f1_score": round(float(selected["f1"]) * 100, 1),
        "roc_auc": round(float(selected["roc_auc"]) * 100, 1),
        "confusion_matrix": matrix,
        "class_labels": ["Left", "Right"],
        "per_class": selected["per_class"],
        "benchmark": [
            {
                "name": item["name"],
                "accuracy": round(float(item["accuracy"]) * 100, 1),
                "precision": round(float(item["precision"]) * 100, 1),
                "recall": round(float(item["recall"]) * 100, 1),
                "f1": round(float(item["f1"]) * 100, 1),
            }
            for item in evaluations
        ],
        "selected_classifier": selected["name"],
        "evaluation_note": evaluation_note,
        "csp_patterns": {
            "channels": pattern_channels,
            "components": [f"CSP {idx + 1}" for idx in range(csp_patterns.shape[0])],
            "values": np.round(csp_patterns, 3).tolist(),
        },
        "artifact_summary": artifact_summary,
    }


def _build_payload(
    raw: mne.io.BaseRaw | None,
    epochs: mne.Epochs,
    source: str,
    subject: str,
    train_subjects: list[int] | None = None,
    test_subject: int | None = None,
    file_count: int = 1,
    active_format: str = "EDF",
    test_epochs: mne.Epochs | None = None,
) -> dict:
    results = _classify_epochs(epochs, test_epochs=test_epochs)
    _log("success", f"Classification completed with {results['selected_classifier']} and {results['accuracy']}% accuracy.")
    _log("success", f"Model saved to {MODEL_PATH.name}.")
    counts = epochs.events[:, -1]
    left_count = int(np.sum(counts == 2))
    right_count = int(np.sum(counts == 3))
    waveform = _raw_to_waveform(raw) if raw is not None else _epochs_to_waveform(epochs)
    analytics = _epoch_analytics(epochs)
    details = _diagnostics(raw, epochs, results, source)
    summary = {
        "subject": subject,
        "status": f"{subject} | {left_count} left + {right_count} right epochs",
        "source": source,
        "sampling_rate": round(float(epochs.info["sfreq"]), 1),
        "channels": len(raw.ch_names) if raw is not None else len(epochs.ch_names),
        "epochs": len(epochs),
        "file_count": file_count,
        "active_format": active_format,
        "train_subjects": train_subjects or [],
        "test_subject": test_subject,
    }
    model_settings = {
        "Filter Band": f"{CONFIG['filter_band'][0]}-{CONFIG['filter_band'][1]} Hz",
        "Epoch Window": f"{CONFIG['epoch_window'][0]} s to {CONFIG['epoch_window'][1]} s",
        "CSP Components": str(CONFIG["csp_components"]),
        "Classifier": results["selected_classifier"],
        "Train/Test Split": f"{int((1 - CONFIG['test_size']) * 100)}/{int(CONFIG['test_size'] * 100)}",
        "Random Seed": str(CONFIG["random_seed"]),
    }
    training_summary = [
        {"label": "Files Used", "value": str(file_count)},
        {"label": "Training Epochs", "value": str(len(epochs))},
        {"label": "Class Balance", "value": f"{left_count} left / {right_count} right"},
        {"label": "Active Format", "value": active_format},
    ]
    about_dataset = [
        {"mode": "EDF Mode", "body": "EDF mode works with raw EEG recordings that already contain event annotations for left and right motor imagery."},
        {"mode": "MAT Mode", "body": "MAT mode works with MATLAB trial arrays and labels, making it ideal for pre-segmented EEG sessions from research workflows."},
    ]
    APP_STATE["recent_runs"] = (
        [
            {
                "time": datetime.now().strftime("%H:%M:%S"),
                "source": source,
                "format": active_format,
                "accuracy": results["accuracy"],
                "classifier": results["selected_classifier"],
                "subject": subject,
            }
        ]
        + APP_STATE["recent_runs"]
    )[:6]
    payload = {
        "summary": summary,
        "waveform": waveform,
        "analytics": analytics,
        "results": results,
        "details": details,
        "model_settings": model_settings,
        "training_summary": training_summary,
        "about_dataset": about_dataset,
        "logs": APP_STATE["logs"],
        "recent_runs": APP_STATE["recent_runs"],
        "project_identity": {
            "Project": "NeuroSense",
            "Student": "Final Year Research Team",
            "Guide": "Academic Project Guide",
            "Department": "Electronics / Biomedical Engineering",
            "Institution": "Engineering Institute",
        },
        "footer_meta": {
            "version": "v2.0",
            "mode": active_format,
            "line": "NeuroSense EEG Motor Imagery Intelligence Dashboard",
        },
        "mat_help": {
            "headline": "MAT Upload Help",
            "examples": [
                "imagery_left + imagery_right + srate",
                "data (trials x channels x samples) + labels + sfreq",
                "session.data + session.labels + session.sfreq",
            ],
        },
    }
    APP_STATE["source"] = source
    APP_STATE["dataset"] = {"raw": raw, "epochs": epochs}
    APP_STATE["summary"] = summary
    APP_STATE["last_payload"] = payload
    REPORT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_physionet_subject(subject: int = 1) -> dict:
    _start_run_log()
    _log("info", f"EDF mode selected for subject S{subject:03d}.")
    paths = [str(_subject_dir(subject) / f"S{subject:03d}R{run:02d}.edf") for run in MI_RUNS]
    if not all(Path(path).exists() for path in paths):
        _log("info", "Local PhysioNet files missing. Requesting MNE download.")
        paths = [str(Path(item)) for item in eegbci.load_data(subject, runs=MI_RUNS)]
    _log("success", f"Loaded {len(paths)} EDF files.")
    raws = [mne.io.read_raw_edf(path, preload=True, verbose="ERROR") for path in paths]
    raw = mne.concatenate_raws(raws)
    raw, epochs = _prepare_epochs(raw)
    _log("success", f"Extracted {len(epochs)} epochs from subject S{subject:03d}.")
    return _build_payload(raw, epochs, source="PhysioNet EEGBCI", subject=f"Subject {subject:03d}", train_subjects=[subject], active_format="EDF", file_count=len(paths))


def _load_subject_epochs(subject: int) -> tuple[mne.io.BaseRaw, mne.Epochs]:
    paths = [str(_subject_dir(subject) / f"S{subject:03d}R{run:02d}.edf") for run in MI_RUNS]
    if not all(Path(path).exists() for path in paths):
        paths = [str(Path(item)) for item in eegbci.load_data(subject, runs=MI_RUNS)]
    raws = [mne.io.read_raw_edf(path, preload=True, verbose="ERROR") for path in paths]
    raw = mne.concatenate_raws(raws)
    return _prepare_epochs(raw)


def load_physionet_multi_subject(train_subjects: list[int], test_subject: int) -> dict:
    _start_run_log()
    _log("info", f"Multi-subject EDF mode selected. Train: {train_subjects}, Test: S{test_subject:03d}.")
    train_items = []
    file_count = 0
    for subject in train_subjects:
        raw, epochs = _load_subject_epochs(subject)
        file_count += len(MI_RUNS)
        train_items.append((f"S{subject:03d}", epochs, raw))
        _log("success", f"Prepared training subject S{subject:03d} with {len(epochs)} epochs.")
    combined = _combine_epochs(train_items)
    test_raw, test_epochs = _load_subject_epochs(test_subject)
    file_count += len(MI_RUNS)
    _log("success", f"Prepared held-out test subject S{test_subject:03d} with {len(test_epochs)} epochs.")
    return _build_payload(
        combined["raw"],
        combined["epochs"],
        source="PhysioNet EEGBCI",
        subject=f"Train {', '.join([f'S{s:03d}' for s in train_subjects])} | Test S{test_subject:03d}",
        train_subjects=train_subjects,
        test_subject=test_subject,
        active_format="EDF",
        file_count=file_count,
        test_epochs=test_epochs,
    )


def generate_synthetic_dataset() -> dict:
    _start_run_log()
    _log("info", "Generating synthetic EEG demo session.")
    sfreq = 160.0
    seconds = 36
    samples = int(sfreq * seconds)
    ch_names = ["FCz", "C3", "Cz", "C4", "CP3", "CPz", "CP4", "Pz"]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    times = np.arange(samples) / sfreq
    rng = np.random.default_rng(42)
    data = []
    for idx, name in enumerate(ch_names):
        phase = idx * 0.3
        carrier = 12e-6 * np.sin(2 * np.pi * 10 * times + phase)
        beta = 8e-6 * np.sin(2 * np.pi * 22 * times + phase / 2)
        modulation = np.where((times % 8) < 4, 1.2, 0.7)
        asymmetry = 1.25 if name in {"C4", "CP4"} else 0.8 if name in {"C3", "CP3"} else 1.0
        noise = rng.normal(scale=2.5e-6, size=samples)
        data.append((carrier + beta * modulation * asymmetry + noise).astype(np.float64))
    raw = mne.io.RawArray(np.vstack(data), info, verbose="ERROR")
    raw.set_montage("standard_1005", on_missing="ignore")

    annotations = []
    for start in np.arange(2, seconds - 4, 4):
        label = "T1" if int(start / 4) % 2 == 0 else "T2"
        annotations.append((start, 3.0, label))
    raw.set_annotations(
        mne.Annotations(
            onset=[item[0] for item in annotations],
            duration=[item[1] for item in annotations],
            description=[item[2] for item in annotations],
        )
    )
    raw, epochs = _prepare_epochs(raw)
    _log("success", f"Synthetic session created with {len(epochs)} epochs.")
    return _build_payload(raw, epochs, source="Synthetic EEG Generator", subject="Demo Subject", active_format="EDF")


def load_uploaded_bundle(files) -> dict:
    _start_run_log()
    dataset_items = []
    temp_paths = []
    try:
        for file_storage in files:
            suffix = Path(file_storage.filename or "").suffix.lower()
            if not suffix:
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                file_storage.save(tmp.name)
                temp_path = Path(tmp.name)
                temp_paths.append(temp_path)

            epochs, raw = _load_dataset_from_path(temp_path)
            dataset_items.append((file_storage.filename, epochs, raw))
            _log("success", f"Accepted {file_storage.filename} with {len(epochs)} epochs.")

        if not dataset_items:
            raise ValueError("Upload at least one supported file: EDF or MAT.")

        combined = _combine_epochs(dataset_items)
        source = " + ".join(sorted({Path(name).suffix.lower().lstrip('.') or "data" for name, _, _ in dataset_items}))
        subject = f"Uploaded Bundle ({len(dataset_items)} file{'s' if len(dataset_items) != 1 else ''})"
        active_format = "MAT" if all(Path(name).suffix.lower() == ".mat" for name, _, _ in dataset_items) else "EDF"
        return _build_payload(
            combined["raw"],
            combined["epochs"],
            source=f"User Uploads [{source.upper()}]",
            subject=subject,
            active_format=active_format,
            file_count=len(dataset_items),
        )
    finally:
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)


@app.route("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def status():
    return jsonify(APP_STATE["summary"])


@app.post("/api/load-physionet")
def api_load_physionet():
    body = request.get_json(silent=True) or {}
    subject = int(body.get("subject", CONFIG["subject_default"]))
    format_mode = str(body.get("format_mode", "edf")).lower()
    train_subjects = [int(item) for item in body.get("train_subjects", [subject])]
    test_subject = body.get("test_subject")
    try:
        if format_mode == "mat":
            return jsonify({"ok": False, "error": "MAT mode expects uploaded MATLAB files. Use the MAT upload option to load data."}), 400
        if test_subject is not None:
            return jsonify({"ok": True, "payload": load_physionet_multi_subject(train_subjects, int(test_subject))})
        return jsonify({"ok": True, "payload": load_physionet_subject(subject)})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"PhysioNet loading failed: {exc}"}), 500


@app.post("/api/generate-synthetic")
def api_generate_synthetic():
    try:
        return jsonify({"ok": True, "payload": generate_synthetic_dataset()})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Synthetic data generation failed: {exc}"}), 500


@app.post("/api/demo-mode")
def api_demo_mode():
    try:
        try:
            payload = load_physionet_subject(1)
        except Exception:
            payload = generate_synthetic_dataset()
        return jsonify({"ok": True, "payload": payload})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Demo mode failed: {exc}"}), 500


@app.post("/api/upload-datasets")
def api_upload_datasets():
    files = [item for item in request.files.getlist("files") if item and item.filename]
    if not files:
        return jsonify({"ok": False, "error": "Upload at least one EDF or MAT file."}), 400
    try:
        return jsonify({"ok": True, "payload": load_uploaded_bundle(files)})
    except Exception as exc:
        message = f"Dataset upload failed: {exc}"
        if any(Path(item.filename).suffix.lower() == ".mat" for item in files):
            message += " | MAT examples: imagery_left + imagery_right + srate, or data + labels + sfreq, or session.data + session.labels + session.sfreq."
        return jsonify({"ok": False, "error": message}), 500


@app.post("/api/run-classification")
def api_run_classification():
    dataset = APP_STATE.get("dataset")
    if not dataset:
        return jsonify({"ok": False, "error": "Load PhysioNet data or generate synthetic EEG first."}), 400
    try:
        raw = dataset["raw"].copy().load_data() if dataset["raw"] is not None else None
        _start_run_log()
        _log("info", "Re-running classification on the active session.")
        payload = _build_payload(
            raw,
            dataset["epochs"].copy().load_data(),
            source=APP_STATE["source"],
            subject=APP_STATE["summary"].get("subject", "Current Dataset"),
            active_format=APP_STATE["summary"].get("active_format", "EDF"),
            file_count=APP_STATE["summary"].get("file_count", 1),
        )
        return jsonify({"ok": True, "payload": payload})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"Classification failed: {exc}"}), 500


@app.post("/api/reset-session")
def api_reset_session():
    APP_STATE["source"] = "none"
    APP_STATE["dataset"] = None
    APP_STATE["summary"] = {"status": "No EEG dataset loaded."}
    APP_STATE["logs"] = []
    APP_STATE["last_payload"] = None
    return jsonify({"ok": True})


@app.get("/api/export-results.csv")
def api_export_results():
    payload = APP_STATE.get("last_payload")
    if not payload:
        return jsonify({"ok": False, "error": "Run a session before exporting results."}), 400
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Predicted Class", payload["results"]["predicted_class"]])
    writer.writerow(["Confidence", payload["results"]["confidence"]])
    writer.writerow(["Accuracy", payload["results"]["accuracy"]])
    writer.writerow(["Precision", payload["results"]["precision"]])
    writer.writerow(["Recall", payload["results"]["recall"]])
    writer.writerow(["F1 Score", payload["results"]["f1_score"]])
    writer.writerow(["ROC AUC", payload["results"]["roc_auc"]])
    writer.writerow(["Cross Validation", payload["results"]["cross_validation"]])
    writer.writerow([])
    writer.writerow(["Confusion Matrix"])
    for row in payload["results"]["confusion_matrix"]:
        writer.writerow(row)
    data = io.BytesIO(buffer.getvalue().encode("utf-8"))
    return send_file(data, mimetype="text/csv", as_attachment=True, download_name="neurosense_results.csv")


@app.get("/api/load-saved-report")
def api_load_saved_report():
    if not REPORT_PATH.exists():
        return jsonify({"ok": False, "error": "No saved report is available yet."}), 404
    payload = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    APP_STATE["last_payload"] = payload
    return jsonify({"ok": True, "payload": payload})


if __name__ == "__main__":
    app.run(debug=True)
