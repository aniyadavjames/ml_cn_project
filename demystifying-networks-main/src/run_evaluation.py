import argparse
import os
import pickle
from pathlib import Path

import numpy as np

from flow_feature_alignment import load_aligned_flow_features
from intrinsic_evaluation import (
    DatasetInput,
    IntrinsicEvaluationFramework,
    PerturbationEvaluationInput,
    SynthDatasetEmbeddings,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--old_model_embeddings_dir",
        type=str,
        default=None,
        help="Directory containing `*_emb.pkl` files for the old model.",
    )
    parser.add_argument(
        "--new_model_embeddings_dir",
        type=str,
        default=None,
        help="Directory containing `*_emb.pkl` files for the new model.",
    )
    parser.add_argument(
        "--old_model_path",
        type=str,
        default=None,
        help="Checkpoint path for the old model, used for perturbation sensitivity.",
    )
    parser.add_argument(
        "--new_model_path",
        type=str,
        default=None,
        help="Checkpoint path for the new model, used for perturbation sensitivity.",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        default=None,
        help="Vocabulary path shared by perturbation evaluation runs.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Config path shared by perturbation evaluation runs.",
    )
    parser.add_argument(
        "--perturbation_dataset_path",
        type=str,
        default=None,
        help="TSV dataset used for perturbation sensitivity evaluation.",
    )
    parser.add_argument(
        "--spm_model_path",
        type=str,
        default=None,
        help="Optional SentencePiece model path for perturbation evaluation.",
    )
    parser.add_argument(
        "--perturbation_batch_size",
        type=int,
        default=64,
        help="Batch size for perturbation sensitivity evaluation.",
    )
    parser.add_argument(
        "--perturbation_seq_length",
        type=int,
        default=128,
        help="Sequence length for perturbation sensitivity evaluation.",
    )
    parser.add_argument(
        "--perturbation_device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for perturbation sensitivity evaluation.",
    )
    parser.add_argument(
        "--perturbation_max_samples",
        type=int,
        default=1024,
        help="Maximum number of samples used for perturbation sensitivity.",
    )
    parser.add_argument(
        "--perturbation_seed",
        type=int,
        default=7,
        help="Random seed for perturbation sensitivity evaluation.",
    )
    parser.add_argument(
        "--auto_cic_dataset_root",
        type=str,
        default=None,
        help=(
            "Dataset root containing `dataset.json`, `picked_file_record`, and "
            "`dataset/x_datagram_{split}.npy` for automatic CKA feature extraction."
        ),
    )
    parser.add_argument(
        "--auto_cic_cache_dir",
        type=str,
        default=None,
        help="Optional cache directory for auto-generated flow features.",
    )
    parser.add_argument(
        "--auto_cic_num_workers",
        type=int,
        default=None,
        help="Worker count for automatic flow feature extraction.",
    )
    parser.add_argument(
        "--auto_cic_max_samples",
        type=int,
        default=None,
        help="Optional sample cap used only for CKA feature alignment.",
    )
    parser.add_argument(
        "--auto_cic_seed",
        type=int,
        default=7,
        help="Random seed used when subsampling automatic CKA features.",
    )
    return parser.parse_args()


def ensure_numpy(embeddings):
    if hasattr(embeddings, "detach"):
        embeddings = embeddings.detach().cpu().numpy()
    return np.asarray(embeddings)


def load_embeddings(path: Path):
    if not path.exists():
        print(f"Warning: embedding file not found at {path}. Skipping.")
        return None, None

    with path.open("rb") as handle:
        embeddings, labels = pickle.load(handle)

    return ensure_numpy(embeddings), labels


def is_synth_dataset(dataset_name: str) -> bool:
    lowered = dataset_name.lower()
    return lowered == "synth" or lowered == "perf" or "synth" in lowered or "perf" in lowered


def discover_embedding_files(embeddings_dir: str):
    if embeddings_dir is None:
        return {}, {}

    base_dir = Path(embeddings_dir).expanduser().resolve()
    if not base_dir.exists():
        print(f"Warning: embeddings directory not found at {base_dir}. Skipping.")
        return {}, {}

    standard_paths = {}
    synth_paths = {}

    for path in sorted(base_dir.glob("*_emb.pkl")):
        dataset_name = path.name[: -len("_emb.pkl")]
        if is_synth_dataset(dataset_name):
            synth_paths[dataset_name] = path
        else:
            standard_paths[dataset_name] = path

    if not standard_paths and not synth_paths:
        print(f"Warning: no `*_emb.pkl` files found in {base_dir}.")

    return standard_paths, synth_paths


def default_synth_label_mapping(raw_label):
    label = str(raw_label)
    filename = os.path.basename(label)
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) > 1:
        return "_".join(parts[:-1])
    return stem


def infer_split_from_dataset_name(dataset_name: str) -> str | None:
    lowered = dataset_name.lower()
    for split_name in ("train", "valid", "test"):
        if split_name in lowered:
            return split_name
    return None


def build_auto_cic_payload(
    dataset_name: str,
    embeddings: np.ndarray,
    args: argparse.Namespace,
    flow_feature_cache: dict[str, tuple[np.ndarray, list[str], list[str], np.ndarray | None]],
):
    if args.auto_cic_dataset_root is None:
        return None

    split_name = infer_split_from_dataset_name(dataset_name)
    if split_name is None:
        print(
            f"Warning: unable to infer train/valid/test split from `{dataset_name}`; "
            "skipping automatic CKA feature extraction."
        )
        return None

    if split_name not in flow_feature_cache:
        flow_feature_cache[split_name] = load_aligned_flow_features(
            args.auto_cic_dataset_root,
            split_name,
            cache_dir=args.auto_cic_cache_dir,
            num_workers=args.auto_cic_num_workers,
            max_samples=args.auto_cic_max_samples,
            seed=args.auto_cic_seed,
            verbose=True,
        )

    flow_features, feature_names, _, selected_indices = flow_feature_cache[split_name]
    expected_rows = len(selected_indices) if selected_indices is not None else flow_features.shape[0]
    if embeddings.shape[0] != expected_rows and selected_indices is None:
        print(
            f"Warning: `{dataset_name}` embeddings have {embeddings.shape[0]} rows but "
            f"auto-generated {split_name} flow features have {flow_features.shape[0]} rows. "
            "Skipping CKA for this dataset."
        )
        return None

    cic_embeddings = embeddings
    if selected_indices is not None:
        cic_embeddings = embeddings[selected_indices]
    elif embeddings.shape[0] != flow_features.shape[0]:
        print(
            f"Warning: `{dataset_name}` embeddings have {embeddings.shape[0]} rows but "
            f"auto-generated {split_name} flow features have {flow_features.shape[0]} rows. "
            "Skipping CKA for this dataset."
        )
        return None

    return {
        "cic_embeddings": cic_embeddings,
        "cic_features": flow_features,
        "cic_feature_names": feature_names,
    }


def build_perturbation_input(
    model_name: str,
    model_path: str | None,
    args: argparse.Namespace,
):
    if model_path is None:
        return None

    missing = []
    for attr_name in ("vocab_path", "config_path", "perturbation_dataset_path"):
        if getattr(args, attr_name) is None:
            missing.append(attr_name)

    if missing:
        print(
            f"Warning: skipping perturbation sensitivity for {model_name}; "
            f"missing {', '.join(missing)}."
        )
        return None

    dataset_name = Path(args.perturbation_dataset_path).stem
    return PerturbationEvaluationInput(
        dataset_name=dataset_name,
        pretrained_model_path=model_path,
        vocab_path=args.vocab_path,
        config_path=args.config_path,
        dataset_path=args.perturbation_dataset_path,
        spm_model_path=args.spm_model_path,
        seq_length=args.perturbation_seq_length,
        batch_size=args.perturbation_batch_size,
        device=args.perturbation_device,
        seed=args.perturbation_seed,
        max_samples=args.perturbation_max_samples,
    )


def run_evaluation_for_model(
    model_name,
    embedding_paths,
    synth_embedding_paths,
    framework,
    perturbation_input=None,
    args=None,
    flow_feature_cache=None,
):
    print("\n" + "=" * 20 + f" EVALUATING: {model_name} " + "=" * 20)

    dataset_inputs = []
    for dataset_name, emb_path in embedding_paths.items():
        embeddings, _ = load_embeddings(emb_path)
        if embeddings is not None:
            dataset_kwargs = {}
            if args is not None and flow_feature_cache is not None:
                auto_cic_payload = build_auto_cic_payload(
                    dataset_name,
                    embeddings,
                    args,
                    flow_feature_cache,
                )
                if auto_cic_payload is not None:
                    dataset_kwargs.update(auto_cic_payload)
            dataset_inputs.append(
                DatasetInput(
                    dataset_name=dataset_name,
                    embeddings=embeddings,
                    **dataset_kwargs,
                )
            )

    synth_dataset_inputs = []
    for dataset_name, emb_path in synth_embedding_paths.items():
        embeddings, labels = load_embeddings(emb_path)
        if embeddings is not None and labels is not None:
            synth_dataset_inputs.append(
                SynthDatasetEmbeddings(
                    dataset_name=dataset_name,
                    embeddings=embeddings,
                    labels=[str(label) for label in labels],
                    label_mapping=default_synth_label_mapping,
                )
            )

    if not dataset_inputs and not synth_dataset_inputs and perturbation_input is None:
        print(f"No valid embedding files found for {model_name}.")
        return

    perturbation_inputs = [perturbation_input] if perturbation_input is not None else None
    results = framework.evaluate(
        dataset_inputs,
        synth_dataset_inputs,
        perturbation_inputs=perturbation_inputs,
    )
    print(results.summary())


def main():
    args = parse_args()
    if (
        not args.old_model_embeddings_dir
        and not args.new_model_embeddings_dir
        and not args.old_model_path
        and not args.new_model_path
    ):
        raise SystemExit(
            "Provide at least one embeddings directory or one model checkpoint path."
        )

    try:
        from dadapy.data import Data  # noqa: F401

        use_id = True
    except ImportError:
        print("Warning: `dadapy` not found. Skipping intrinsic dimensionality.")
        use_id = False

    old_perturbation_input = build_perturbation_input(
        "Old Pre-trained Model",
        args.old_model_path,
        args,
    )
    new_perturbation_input = build_perturbation_input(
        "New Generated Model",
        args.new_model_path,
        args,
    )
    flow_feature_cache = {}

    framework = IntrinsicEvaluationFramework(
        compute_anisotropy=True,
        compute_intrinsic_dim=use_id,
        compute_cka_cic=bool(args.auto_cic_dataset_root),
        compute_causal_sensitivity=True,
        compute_synth_math=True,
        compute_perturbation=bool(old_perturbation_input or new_perturbation_input),
        verbose=False,
    )

    old_embedding_paths, old_synth_paths = discover_embedding_files(
        args.old_model_embeddings_dir
    )
    new_embedding_paths, new_synth_paths = discover_embedding_files(
        args.new_model_embeddings_dir
    )

    if args.old_model_embeddings_dir:
        run_evaluation_for_model(
            "Old Pre-trained Model",
            old_embedding_paths,
            old_synth_paths,
            framework,
            perturbation_input=old_perturbation_input,
            args=args,
            flow_feature_cache=flow_feature_cache,
        )
    elif old_perturbation_input is not None:
        run_evaluation_for_model(
            "Old Pre-trained Model",
            {},
            {},
            framework,
            perturbation_input=old_perturbation_input,
            args=args,
            flow_feature_cache=flow_feature_cache,
        )

    if args.new_model_embeddings_dir:
        run_evaluation_for_model(
            "New Generated Model",
            new_embedding_paths,
            new_synth_paths,
            framework,
            perturbation_input=new_perturbation_input,
            args=args,
            flow_feature_cache=flow_feature_cache,
        )
    elif new_perturbation_input is not None:
        run_evaluation_for_model(
            "New Generated Model",
            {},
            {},
            framework,
            perturbation_input=new_perturbation_input,
            args=args,
            flow_feature_cache=flow_feature_cache,
        )


if __name__ == "__main__":
    main()
