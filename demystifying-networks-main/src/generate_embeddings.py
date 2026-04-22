import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
ET_BERT_ROOT = (SCRIPT_DIR.parent.parent / "ET-BERT").resolve()
if str(ET_BERT_ROOT) not in sys.path:
    sys.path.insert(0, str(ET_BERT_ROOT))

from finetuning.run_classifier import Classifier, batch_loader
from uer.opts import model_opts
from uer.utils import str2tokenizer
from uer.utils.config import load_hyperparam
from uer.utils.constants import CLS_TOKEN, SEP_TOKEN
from uer.utils.seed import set_seed


def resolve_path(path_str: str) -> Path:
    return Path(path_str).expanduser().resolve()


def build_arg_parser(require_output_path: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        required=True,
        help="Path to the ET-BERT checkpoint (.bin).",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        required=True,
        help="Path to the ET-BERT vocabulary file.",
    )
    parser.add_argument(
        "--spm_model_path",
        type=str,
        default=None,
        help="Optional SentencePiece model path. Leave unset for the default vocab tokenizer.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=str(ET_BERT_ROOT / "bert_base_config.json"),
        help="Path to the ET-BERT config JSON.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the TSV dataset used for embedding extraction.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=require_output_path,
        default=None,
        help="Where to save the output pickle `(embeddings, labels)`.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size used for embedding extraction.",
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        default=128,
        help="Sequence length expected by ET-BERT.",
    )
    parser.add_argument(
        "--pooling",
        choices=["mean", "max", "first", "last"],
        default="first",
        help="Pooling strategy used for the encoder output.",
    )
    parser.add_argument(
        "--tokenizer",
        choices=["bert", "char", "space"],
        default="bert",
        help="Tokenizer used by ET-BERT.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device used for inference.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--soft_targets",
        action="store_true",
        help="Kept for compatibility with ET-BERT args; unused for embedding export.",
    )
    parser.add_argument(
        "--soft_alpha",
        type=float,
        default=0.5,
        help="Kept for compatibility with ET-BERT args; unused for embedding export.",
    )

    model_opts(parser)
    return parser


def normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    args.pretrained_model_path = str(resolve_path(args.pretrained_model_path))
    args.vocab_path = str(resolve_path(args.vocab_path))
    args.config_path = str(resolve_path(args.config_path))
    args.dataset_path = str(resolve_path(args.dataset_path))
    if getattr(args, "output_path", None):
        args.output_path = str(resolve_path(args.output_path))
    return load_hyperparam(args)


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser(require_output_path=True)
    return normalize_args(parser.parse_args())


def build_runtime_args(
    pretrained_model_path: str,
    vocab_path: str,
    config_path: str,
    dataset_path: str,
    *,
    output_path: str | None = None,
    spm_model_path: str | None = None,
    batch_size: int = 64,
    seq_length: int = 128,
    pooling: str = "first",
    tokenizer: str = "bert",
    device: str = "auto",
    seed: int = 7,
) -> argparse.Namespace:
    parser = build_arg_parser(require_output_path=False)
    arg_list = [
        "--pretrained_model_path",
        pretrained_model_path,
        "--vocab_path",
        vocab_path,
        "--config_path",
        config_path,
        "--dataset_path",
        dataset_path,
        "--batch_size",
        str(batch_size),
        "--seq_length",
        str(seq_length),
        "--pooling",
        pooling,
        "--tokenizer",
        tokenizer,
        "--device",
        device,
        "--seed",
        str(seed),
    ]
    if output_path is not None:
        arg_list.extend(["--output_path", output_path])
    if spm_model_path is not None:
        arg_list.extend(["--spm_model_path", spm_model_path])
    return normalize_args(parser.parse_args(arg_list))


def choose_device(requested_device: str) -> torch.device:
    if requested_device == "cpu":
        return torch.device("cpu")
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("`--device cuda` was requested, but CUDA is not available.")
        return torch.device("cuda:0")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def read_embedding_dataset(args: argparse.Namespace):
    dataset = []
    labels = []
    dataset_path = Path(args.dataset_path)

    with dataset_path.open("r", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        columns = {name: index for index, name in enumerate(header)}

        if "text_a" not in columns:
            raise ValueError(
                f"Dataset at {dataset_path} must contain a `text_a` column."
            )

        for row_index, raw_line in enumerate(handle):
            line = raw_line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            text_a = parts[columns["text_a"]]

            if "text_b" not in columns:
                src = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + args.tokenizer.tokenize(text_a)
                )
                seg = [1] * len(src)
            else:
                text_b = parts[columns["text_b"]]
                src_a = args.tokenizer.convert_tokens_to_ids(
                    [CLS_TOKEN] + args.tokenizer.tokenize(text_a) + [SEP_TOKEN]
                )
                src_b = args.tokenizer.convert_tokens_to_ids(
                    args.tokenizer.tokenize(text_b) + [SEP_TOKEN]
                )
                src = src_a + src_b
                seg = [1] * len(src_a) + [2] * len(src_b)

            if len(src) > args.seq_length:
                src = src[: args.seq_length]
                seg = seg[: args.seq_length]
            while len(src) < args.seq_length:
                src.append(0)
                seg.append(0)

            dataset.append((src, 0, seg))
            labels.append(extract_label(parts, columns, row_index))

    if not dataset:
        raise ValueError(f"No usable rows found in dataset: {dataset_path}")

    return dataset, labels


def extract_label(parts, columns, row_index: int):
    for key in ("label", "filename", "file", "id", "sample_id", "name"):
        if key in columns and columns[key] < len(parts):
            return parts[columns[key]]
    return f"sample_{row_index}"


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict):
        for candidate_key in ("state_dict", "model_state_dict", "model"):
            candidate = checkpoint.get(candidate_key)
            if isinstance(candidate, dict):
                checkpoint = candidate
                break

    if not isinstance(checkpoint, dict):
        raise TypeError(
            f"Unsupported checkpoint format in {checkpoint_path}: {type(checkpoint)!r}"
        )

    model_state = model.state_dict()
    compatible_state = {}
    skipped_keys = []

    for key, value in checkpoint.items():
        clean_key = key[7:] if key.startswith("module.") else key
        if clean_key in model_state and model_state[clean_key].shape == value.shape:
            compatible_state[clean_key] = value
        else:
            skipped_keys.append(clean_key)

    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)

    print(
        "Loaded checkpoint weights:",
        f"{len(compatible_state)} matched,",
        f"{len(skipped_keys)} skipped,",
        f"{len(missing_keys)} missing after load,",
        f"{len(unexpected_keys)} unexpected.",
    )


def build_model(args: argparse.Namespace) -> Classifier:
    args.labels_num = 1
    if isinstance(args.tokenizer, str):
        args.tokenizer = str2tokenizer[args.tokenizer](args)
    args.device = choose_device(args.device)

    model = Classifier(args)
    load_checkpoint(model, args.pretrained_model_path)
    model = model.to(args.device)
    model.eval()
    return model


def encode_batch(
    model: Classifier,
    src_batch: torch.Tensor,
    seg_batch: torch.Tensor,
    device: torch.device,
    pooling: str,
) -> torch.Tensor:
    with torch.no_grad():
        src_batch = src_batch.to(device)
        seg_batch = seg_batch.to(device)
        hidden = model.embedding(src_batch, seg_batch)
        hidden = model.encoder(hidden, seg_batch)

        if pooling == "mean":
            pooled = hidden.mean(dim=1)
        elif pooling == "max":
            pooled = hidden.max(dim=1).values
        elif pooling == "last":
            pooled = hidden[:, -1, :]
        else:
            pooled = hidden[:, 0, :]

    return pooled.cpu()


def generate_embeddings(args: argparse.Namespace):
    if isinstance(args.tokenizer, str):
        args.tokenizer = str2tokenizer[args.tokenizer](args)
    dataset, labels = read_embedding_dataset(args)
    src = torch.LongTensor([example[0] for example in dataset])
    tgt = torch.zeros(len(dataset), dtype=torch.long)
    seg = torch.LongTensor([example[2] for example in dataset])

    model = build_model(args)

    batches = []
    loader = batch_loader(args.batch_size, src, tgt, seg, None)
    for src_batch, _, seg_batch, _ in tqdm(loader, desc="Encoding", total=(len(dataset) + args.batch_size - 1) // args.batch_size):
        batches.append(
            encode_batch(
                model=model,
                src_batch=src_batch,
                seg_batch=seg_batch,
                device=args.device,
                pooling=args.pooling,
            )
        )

    embeddings = torch.cat(batches, dim=0).numpy().astype(np.float32, copy=False)
    return embeddings, labels


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"Using ET-BERT root: {ET_BERT_ROOT}")
    print(f"Using device: {choose_device(args.device)}")
    print(f"Reading dataset: {args.dataset_path}")

    embeddings, labels = generate_embeddings(args)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump((embeddings, labels), handle)

    print(
        f"Saved {embeddings.shape[0]} embeddings with shape {embeddings.shape} "
        f"to {output_path}"
    )


if __name__ == "__main__":
    main()
