"""Fine-tunes one encoder candidate, then exports it to INT8 ONNX. NOT yet run: see docs/MODEL_CARD.md.

Run it on a GPU notebook that imports this module and holds no logic of its own. It refuses template
fixtures unless --allow-fixture is passed, because fixture training says nothing about real performance.
Early stopping uses dev mean misroute cost, not loss.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from switchboard_inference.data.splits import load_split
from switchboard_inference.decision import CANONICAL_INTENTS, CostMatrix, decide, load_tenants
from switchboard_inference.eval.simulate import scoped

CANDIDATES = {
    "afro-mini": "Davlan/afro-xlmr-mini",
    "afro-small": "Davlan/afro-xlmr-small",
    "e5-small": "intfloat/multilingual-e5-small",
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}


def dev_mean_cost(probabilities: np.ndarray, labels: list[str], rows: list[dict], tenants, matrix) -> float:
    from datetime import datetime

    from switchboard_inference.decision import route_cost

    total = 0.0
    for row, p in zip(rows, probabilities, strict=True):
        tenant = tenants[row["tenant"]]
        scores = scoped(dict(zip(labels, p.tolist(), strict=True)), tenant)
        at = datetime.fromisoformat(row["at"])
        d = decide(tenant, matrix, scores, at, llm_available=False, clarify_turns=2)
        total += matrix.costs["reception"] if d.action == "reception" else route_cost(tenant, matrix, row["intent"], d.department, at)
    return total / len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--overfit", action="store_true", help="overfit 20 examples first, as a sanity check")
    parser.add_argument("--allow-fixture", action="store_true")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    root = Path(args.data)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("reportable") and not args.allow_fixture:
        raise SystemExit("this dataset is not reportable. Pass --allow-fixture only to test the training code.")

    train, dev = load_split(root, "train"), load_split(root, "dev")
    if args.overfit:
        train = train[:20]
    labels = sorted({*CANONICAL_INTENTS, *(r["intent"] for r in train)})
    index = {label: i for i, label in enumerate(labels)}
    name = CANDIDATES[args.candidate]
    tokenizer = AutoTokenizer.from_pretrained(name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    tenants = load_tenants()
    matrix = CostMatrix.load("configs/cost_matrix.yaml")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def batches(rows, shuffle):
        order = np.random.default_rng(0).permutation(len(rows)) if shuffle else np.arange(len(rows))
        for start in range(0, len(rows), args.batch):
            yield [rows[i] for i in order[start : start + args.batch]]

    def encode(rows):
        return tokenizer([r["text"] for r in rows], padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)

    best = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for rows in batches(train, shuffle=True):
            loss = model(**encode(rows), labels=torch.tensor([index[r["intent"]] for r in rows], device=device)).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        outputs = []
        with torch.no_grad():
            for rows in batches(dev, shuffle=False):
                outputs.append(torch.softmax(model(**encode(rows)).logits, dim=-1).cpu().numpy())
        cost = dev_mean_cost(np.concatenate(outputs), labels, dev, tenants, matrix)
        print(json.dumps({"epoch": epoch, "dev_mean_cost": cost}))
        if cost < best:
            best = cost
            model.save_pretrained(out / "best")
            tokenizer.save_pretrained(out / "best")

    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    (out / "labels.json").write_text(json.dumps(labels), encoding="utf-8")
    (out / "run.json").write_text(
        json.dumps({"candidate": name, "git_sha": sha, "dataset": manifest["splits"], "best_dev_mean_cost": best, "args": vars(args)}, indent=1),
        encoding="utf-8",
    )
    export(out)


def export(out: Path) -> None:
    """ONNX, then INT8. Vocabulary trimming and parity checks follow in M5, once a model exists."""
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(out / "best")
    model = AutoModelForSequenceClassification.from_pretrained(out / "best").eval().cpu()
    sample = tokenizer(["sample text"], return_tensors="pt")
    names = list(sample.keys())
    torch.onnx.export(
        model, tuple(sample.values()), out / "fp32.onnx", input_names=names, output_names=["logits"],
        dynamic_axes={n: {0: "batch", 1: "sequence"} for n in names},
    )  # fmt: skip
    quantize_dynamic(out / "fp32.onnx", out / "model.onnx", weight_type=QuantType.QInt8)
    tokenizer.backend_tokenizer.save(str(out / "tokenizer.json"))


if __name__ == "__main__":
    main()
