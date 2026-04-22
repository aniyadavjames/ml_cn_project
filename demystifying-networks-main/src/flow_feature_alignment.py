from __future__ import annotations

from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import math
import os

import numpy as np
from tqdm import tqdm


TLS_EXTENSIONS = [
    "tls.record.content_type",
    "tls.record.opaque_type",
    "tls.handshake.type",
]

FLOW_FEATURE_NAMES = [
    "packet_count",
    "payload_packet_count",
    "flow_duration",
    "ip_bytes_abs_total",
    "ip_bytes_mean_abs",
    "ip_bytes_std_abs",
    "payload_bytes_abs_total",
    "payload_bytes_uplink_total",
    "payload_bytes_downlink_total",
    "payload_bytes_mean_abs",
    "payload_bytes_std_abs",
    "interarrival_mean",
    "interarrival_std",
    "interarrival_max",
    "interarrival_min",
    "ip_uplink_fraction",
    "payload_uplink_fraction",
    "src_port",
    "dst_port",
    "is_tcp",
]

CACHE_VERSION = 1


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_value = _mean(values)
    return float(math.sqrt(sum((value - mean_value) ** 2 for value in values) / len(values)))


def _parse_path_metadata(path: str) -> tuple[float, float, float]:
    basename = os.path.basename(path)
    flow_name = basename.split(".pcap.", 1)[-1]
    flow_name = flow_name.rsplit(".pcap", 1)[0]
    parts = flow_name.split("_")

    if len(parts) < 5:
        return 0.0, 0.0, 0.0

    protocol = parts[0].upper()
    try:
        src_port = float(parts[-3])
    except ValueError:
        src_port = 0.0
    try:
        dst_port = float(parts[-1])
    except ValueError:
        dst_port = 0.0
    return src_port, dst_port, float(protocol == "TCP")


def _protocol_candidates(path: str) -> list[str | None]:
    basename = os.path.basename(path)
    if ".TCP_" in basename:
        return ["tcp", "udp", None]
    if ".UDP_" in basename:
        return ["udp", "tcp", None]
    return ["tcp", "udp", None]


def _extract_first_flow(path: str):
    from flowcontainer.extractor import extract

    for protocol in _protocol_candidates(path):
        kwargs = {}
        if protocol is not None:
            kwargs["filter"] = protocol
        if protocol == "tcp":
            kwargs["extension"] = TLS_EXTENSIONS

        try:
            flows = extract(path, **kwargs)
        except Exception:
            continue

        if flows:
            return next(iter(flows.values()))
    return None


def _extract_feature_row(path: str) -> np.ndarray:
    src_port, dst_port, is_tcp = _parse_path_metadata(path)
    flow = _extract_first_flow(path)

    if flow is None:
        return np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                src_port,
                dst_port,
                is_tcp,
            ],
            dtype=np.float32,
        )

    ip_lengths = [float(value) for value in (getattr(flow, "ip_lengths", []) or [])]
    payload_lengths = [float(value) for value in (getattr(flow, "payload_lengths", []) or [])]
    timestamps = [float(value) for value in (getattr(flow, "ip_timestamps", []) or [])]
    if not timestamps:
        timestamps = [float(value) for value in (getattr(flow, "payload_timestamps", []) or [])]

    abs_ip_lengths = [abs(value) for value in ip_lengths]
    abs_payload_lengths = [abs(value) for value in payload_lengths]
    payload_uplink = [value for value in payload_lengths if value > 0]
    payload_downlink = [-value for value in payload_lengths if value < 0]
    interarrivals = [
        max(0.0, next_ts - current_ts)
        for current_ts, next_ts in zip(timestamps, timestamps[1:])
    ]

    packet_count = float(len(ip_lengths))
    payload_packet_count = float(len(payload_lengths))
    flow_duration = float(
        max(
            0.0,
            float(getattr(flow, "time_end", 0.0)) - float(getattr(flow, "time_start", 0.0)),
        )
    )

    return np.asarray(
        [
            packet_count,
            payload_packet_count,
            flow_duration,
            float(sum(abs_ip_lengths)),
            _mean(abs_ip_lengths),
            _std(abs_ip_lengths),
            float(sum(abs_payload_lengths)),
            float(sum(payload_uplink)),
            float(sum(payload_downlink)),
            _mean(abs_payload_lengths),
            _std(abs_payload_lengths),
            _mean(interarrivals),
            _std(interarrivals),
            float(max(interarrivals)) if interarrivals else 0.0,
            float(min(interarrivals)) if interarrivals else 0.0,
            float(sum(1 for value in ip_lengths if value > 0) / len(ip_lengths)) if ip_lengths else 0.0,
            float(sum(1 for value in payload_lengths if value > 0) / len(payload_lengths)) if payload_lengths else 0.0,
            src_port,
            dst_port,
            is_tcp,
        ],
        dtype=np.float32,
    )


def _extract_feature_rows(
    paths: list[str],
    *,
    num_workers: int | None,
    verbose: bool,
) -> np.ndarray:
    worker_count = num_workers or min(8, os.cpu_count() or 1)
    if worker_count <= 1:
        rows = []
        iterator = paths
        if verbose:
            iterator = tqdm(iterator, desc="Extracting flow features")
        for path in iterator:
            rows.append(_extract_feature_row(path))
        return np.stack(rows)

    rows = []
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        iterator = executor.map(_extract_feature_row, paths, chunksize=32)
        if verbose:
            iterator = tqdm(iterator, total=len(paths), desc="Extracting flow features")
        for row in iterator:
            rows.append(row)
    return np.stack(rows)


def _load_original_records(dataset_root: Path) -> list[tuple[str, int, str]]:
    dataset_json_path = dataset_root / "dataset.json"
    picked_file_path = dataset_root / "picked_file_record"
    if not dataset_json_path.exists():
        raise FileNotFoundError(f"Missing dataset metadata: {dataset_json_path}")
    if not picked_file_path.exists():
        raise FileNotFoundError(f"Missing picked file record: {picked_file_path}")

    with dataset_json_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    with picked_file_path.open("r", encoding="utf-8") as handle:
        picked_paths = [line.rstrip("\n") for line in handle if line.strip()]

    expected_total = sum(int(entry["samples"]) for entry in metadata.values())
    if expected_total != len(picked_paths):
        raise ValueError(
            f"picked_file_record count ({len(picked_paths)}) does not match "
            f"dataset.json samples ({expected_total})."
        )

    cursor = 0
    records = []
    for label in sorted(metadata.keys(), key=lambda value: int(value)):
        sample_count = int(metadata[label]["samples"])
        payloads = metadata[label]["payload"]
        for sample_index in range(1, sample_count + 1):
            payload = str(payloads[str(sample_index)])
            records.append((payload, int(label), picked_paths[cursor]))
            cursor += 1
    return records


def align_split_paths(dataset_root: str | os.PathLike[str], split: str) -> list[str]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    split = split.lower()
    x_path = dataset_root / "dataset" / f"x_datagram_{split}.npy"
    y_path = dataset_root / "dataset" / f"y_{split}.npy"
    if not x_path.exists() or not y_path.exists():
        raise FileNotFoundError(f"Missing split arrays for `{split}` under {dataset_root / 'dataset'}")

    queues: dict[tuple[str, int], deque[str]] = defaultdict(deque)
    for payload, label, path in _load_original_records(dataset_root):
        queues[(payload, label)].append(path)

    payloads = np.load(x_path, allow_pickle=True)
    labels = np.load(y_path, allow_pickle=True)

    aligned_paths = []
    missing = []
    for index, (payload, label) in enumerate(zip(payloads, labels)):
        key = (str(payload), int(label))
        if not queues[key]:
            missing.append(index)
            if len(missing) >= 5:
                break
            continue
        aligned_paths.append(queues[key].popleft())

    if missing:
        raise ValueError(
            f"Failed to align {len(missing)} rows for split `{split}` under {dataset_root}."
        )
    return aligned_paths


def load_aligned_flow_features(
    dataset_root: str | os.PathLike[str],
    split: str,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    num_workers: int | None = None,
    max_samples: int | None = None,
    seed: int = 7,
    verbose: bool = True,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray | None]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    split = split.lower()
    cache_root = Path(cache_dir).expanduser().resolve() if cache_dir else dataset_root / "flow_feature_cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    sample_suffix = ""
    if max_samples is not None and max_samples > 0:
        sample_suffix = f"_n{int(max_samples)}_s{int(seed)}"
    cache_path = cache_root / f"{split}_flow_features_v{CACHE_VERSION}{sample_suffix}.npz"

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=True) as cache:
            selected_indices = None
            if "selected_indices" in cache.files:
                loaded_indices = np.asarray(cache["selected_indices"], dtype=np.int64)
                if loaded_indices.size > 0:
                    selected_indices = loaded_indices
            return (
                np.asarray(cache["features"], dtype=np.float32),
                cache["feature_names"].tolist(),
                cache["paths"].tolist(),
                selected_indices,
            )

    aligned_paths = align_split_paths(dataset_root, split)
    selected_indices = None
    if max_samples is not None and 0 < max_samples < len(aligned_paths):
        rng = np.random.default_rng(seed)
        selected_indices = np.sort(
            rng.choice(len(aligned_paths), size=max_samples, replace=False)
        ).astype(np.int64)
        aligned_paths = [aligned_paths[index] for index in selected_indices]

    features = _extract_feature_rows(
        aligned_paths,
        num_workers=num_workers,
        verbose=verbose,
    )

    np.savez_compressed(
        cache_path,
        features=features,
        feature_names=np.asarray(FLOW_FEATURE_NAMES, dtype=object),
        paths=np.asarray(aligned_paths, dtype=object),
        selected_indices=selected_indices if selected_indices is not None else np.asarray([], dtype=np.int64),
    )
    return features, list(FLOW_FEATURE_NAMES), aligned_paths, selected_indices
