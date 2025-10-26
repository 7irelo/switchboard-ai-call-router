"""ONNX, then INT8, then vocabulary trim, with a parity check at each step (arm E5)."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np


def torch_logits(model_dir: Path, texts: list[str]) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
    out = []
    with torch.no_grad():
        for text in texts:
            out.append(model(**tokenizer([text], truncation=True, max_length=64, return_tensors="pt")).logits[0].numpy())
    return np.stack(out)


def to_onnx(model_dir: Path, out_path: Path) -> None:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval().cpu()
    sample = tokenizer(["sample text for export"], return_tensors="pt")
    names = list(sample.keys())
    torch.onnx.export(
        model, tuple(sample.values()), str(out_path), input_names=names, output_names=["logits"],
        dynamic_axes={n: {0: "batch", 1: "sequence"} for n in names}, opset_version=17, dynamo=False,
    )  # fmt: skip


def quantise(fp32: Path, int8: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)


def keep_ids(tokenizer_json: dict, texts: Iterable[str], tokenizer) -> list[int]:
    """Every token the data actually uses, plus the special tokens. Scans ids, not guesses."""
    used = {i for text in texts for i in tokenizer.encode(text).ids}
    used |= {t["id"] for t in tokenizer_json.get("added_tokens", [])}
    model = tokenizer_json["model"]
    if model.get("unk_id") is not None:
        used.add(model["unk_id"])
    elif model.get("unk_token") in model.get("vocab", {}):
        used.add(model["vocab"][model["unk_token"]])
    return sorted(used)


def trim_vocabulary(model_dir: Path, out_dir: Path, texts: list[str]) -> dict:
    """Keeps only embeddings the data uses. Multilingual vocabularies are most of a small model's size."""
    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForSequenceClassification

    out_dir.mkdir(parents=True, exist_ok=True)
    fast = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    raw = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    keep = keep_ids(raw, texts, fast)
    remap = {old: new for new, old in enumerate(keep)}

    vocab = raw["model"]["vocab"]
    if isinstance(vocab, dict):  # WordPiece
        raw["model"]["vocab"] = {tok: remap[i] for tok, i in vocab.items() if i in remap}
    else:  # Unigram: a list whose position is the id.
        raw["model"]["vocab"] = [vocab[i] for i in keep if i < len(vocab)]
        if raw["model"].get("unk_id") is not None:
            raw["model"]["unk_id"] = remap[raw["model"]["unk_id"]]
    for token in raw.get("added_tokens", []):
        token["id"] = remap[token["id"]]
    post = raw.get("post_processor") or {}
    for processor in [post, *post.get("processors", [])]:
        for special in (processor.get("special_tokens") or {}).values():
            special["ids"] = [remap[i] for i in special["ids"]]
        for key in ("sep", "cls"):
            if isinstance(processor.get(key), list):
                processor[key][1] = remap[processor[key][1]]

    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    embeddings = model.get_input_embeddings()
    new = torch.nn.Embedding(len(keep), embeddings.embedding_dim)
    new.weight.data = embeddings.weight.data[torch.tensor(keep)].clone()
    model.set_input_embeddings(new)
    model.config.vocab_size = len(keep)
    model.save_pretrained(out_dir)
    for name in model_dir.iterdir():
        if name.is_file() and name.suffix in {".json", ".txt", ".model"} and name.name not in {"config.json", "tokenizer.json", "generation_config.json"}:
            shutil.copy2(name, out_dir / name.name)
    (out_dir / "tokenizer.json").write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return {"vocab_before": embeddings.num_embeddings, "vocab_after": len(keep)}


def onnx_logits(path: Path, tokenizer_path: Path, texts: list[str]) -> tuple[np.ndarray, list[float]]:
    import onnxruntime as ort
    from tokenizers import Tokenizer

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    inputs = {i.name for i in session.get_inputs()}
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(64)
    out, times = [], []
    for text in texts:
        e = tokenizer.encode(text)
        feed = {"input_ids": np.array([e.ids], dtype=np.int64), "attention_mask": np.array([e.attention_mask], dtype=np.int64)}
        if "token_type_ids" in inputs:
            feed["token_type_ids"] = np.array([e.type_ids], dtype=np.int64)
        started = time.perf_counter()
        out.append(session.run(None, feed)[0][0])
        times.append((time.perf_counter() - started) * 1000)
    return np.stack(out), times


def describe(reference: np.ndarray, logits: np.ndarray, times: list[float], path: Path) -> dict:
    ordered = sorted(times[5:] or times)
    return {
        "size_mb": round(path.stat().st_size / 1e6, 1),
        "top1_agreement": float((reference.argmax(1) == logits.argmax(1)).mean()),
        "max_logit_diff": float(np.abs(reference - logits).max()),
        "p50_ms": ordered[len(ordered) // 2],
        "p95_ms": ordered[int(0.95 * len(ordered))],
    }


def run(model_dir: Path, out: Path, texts: list[str], reference_texts: list[str]) -> dict:
    report: dict = {"samples": len(reference_texts), "pinned_core": False, "stage_budget_ms": 30}
    reference = torch_logits(model_dir, reference_texts)
    fp32, int8 = out / "fp32.onnx", out / "int8.onnx"
    to_onnx(model_dir, fp32)
    tokenizer = model_dir / "tokenizer.json"
    logits, times = onnx_logits(fp32, tokenizer, reference_texts)
    report["fp32"] = describe(reference, logits, times, fp32)
    quantise(fp32, int8)
    logits, times = onnx_logits(int8, tokenizer, reference_texts)
    report["int8"] = describe(reference, logits, times, int8)

    trimmed = out / "trimmed"
    report["trim"] = trim_vocabulary(model_dir, trimmed, texts)
    trimmed_fp32, trimmed_int8 = out / "trimmed_fp32.onnx", out / "model.onnx"
    to_onnx(trimmed, trimmed_fp32)
    quantise(trimmed_fp32, trimmed_int8)
    logits, times = onnx_logits(trimmed_int8, trimmed / "tokenizer.json", reference_texts)
    report["int8_trimmed"] = describe(reference, logits, times, trimmed_int8)
    shutil.copy2(trimmed / "tokenizer.json", out / "tokenizer.json")
    (out / "export_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def main() -> None:
    from switchboard_inference.data.splits import load_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="training output directory, holding best/")
    parser.add_argument("--data", required=True)
    parser.add_argument("--samples", type=int, default=200)
    args = parser.parse_args()
    root, out = Path(args.data), Path(args.run)
    texts = [r["text"] for split in ("train", "calibration", "dev") for r in load_split(root, split)]
    dev = [r["text"] for r in load_split(root, "dev")][: args.samples]
    print(json.dumps(run(out / "best", out, texts, dev), indent=1))


if __name__ == "__main__":
    main()
