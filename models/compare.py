"""compare_models.py - train several architectures on the SAME split, pick the best on VALIDATION,
save it as best_model.pt, and visualise predictions from all models.

Selection uses the validation split only. Test numbers are reported for information and the
final test score is only quoted for the chosen model.
"""
import json
import os
import random
import shutil

import numpy as np
import torch

from base_model import load_checkpoint
from client import Client, build_transforms, evaluate_checkpoint, make_loader, make_split
from f1engine import F1Engine

DEFAULT_ARCHS = ("resnet18", "mobilenet_v2", "efficientnet_b0")


def train_all(base, archs=DEFAULT_ARCHS, train_dir="Training", test_dir="Testing",
              epochs=10, img_size=128, seed=42, out_dir="/content/results"):
    os.makedirs(out_dir, exist_ok=True)
    split, classes = make_split(base, train_dir, test_dir, seed)   # ONE split shared by all models
    results = {}
    for arch in archs:
        print(f"\n===== Training {arch} =====")
        c = Client(base, arch=arch, img_size=img_size, seed=seed, split=split, classes=classes)
        c.train(epochs)                                            # keeps best epoch by val acc
        path = os.path.join(out_dir, f"{arch}.pt")
        c.save(path)
        val = F1Engine(classes).evaluate(c.model, c.val_loader, c.device)
        _, test = c.evaluate_test()
        results[arch] = {"ckpt": path,
                         "val_acc": val["accuracy"], "val_macro_f1": val["macro_f1"],
                         "test_acc": test["accuracy"], "test_macro_f1": test["macro_f1"],
                         "test_per_class_recall": {k: v["recall"] for k, v in test["per_class"].items()}}
        del c
        torch.cuda.empty_cache()

    best = max(results, key=lambda a: results[a]["val_macro_f1"])   # selection on VAL only
    best_path = os.path.join(out_dir, "best_model.pt")
    shutil.copy(results[best]["ckpt"], best_path)
    with open(os.path.join(out_dir, "comparison.json"), "w") as f:
        json.dump({"selected_by": "val_macro_f1", "best": best, "results": results}, f, indent=2)

    print("\n" + "=" * 68)
    print(f"{'model':16s} {'val_acc':>8s} {'val_F1':>8s} | {'test_acc':>8s} {'test_F1':>8s}")
    for a, r in results.items():
        mark = "  <-- BEST (by val macro-F1)" if a == best else ""
        print(f"{a:16s} {r['val_acc']*100:7.2f}% {r['val_macro_f1']:8.4f} | "
              f"{r['test_acc']*100:7.2f}% {r['test_macro_f1']:8.4f}{mark}")
    print(f"\nSaved best model -> {best_path}")
    return best, best_path, results


def final_report(base, best_path, out_dir="/content/results"):
    """Official test score for the chosen model + confusion matrix."""
    plot = os.path.join(out_dir, "best_model_confusion_matrix.png")
    m = evaluate_checkpoint(base, best_path, plot_path=plot)
    print(f"FINAL TEST accuracy (chosen model): {m['accuracy']*100:.2f}%   macro-F1: {m['macro_f1']:.4f}")
    for c, v in m["per_class"].items():
        print(f"  {c:12s} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f} n={v['support']}")
    return m, plot


@torch.no_grad()
def _probs(model, loader, device):
    model.eval()
    out, ys = [], []
    for x, y in loader:
        out.append(torch.softmax(model(x.to(device)), 1).cpu()); ys += y.tolist()
    return torch.cat(out).numpy(), np.array(ys)


def ensemble_eval(base, ckpt_paths):
    """Accuracy of each model and of the averaged-probability ensemble on the shared test set."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    all_probs, ys, classes = [], None, None
    for name, path in ckpt_paths.items():
        model, ckpt = load_checkpoint(path, device)
        _, ev_tf = build_transforms(ckpt["extra"]["img_size"])
        loader = make_loader(base, ckpt["split"]["test"], ckpt["classes"], ev_tf, 32, False)
        p, y = _probs(model, loader, device)
        all_probs.append(p); ys, classes = y, ckpt["classes"]
        print(f"{name:16s} test acc = {(p.argmax(1) == y).mean()*100:.2f}%")
    ens = np.mean(all_probs, axis=0)
    e = F1Engine(classes); e.update(ys, ens.argmax(1)); m = e.metrics()
    print(f"{'ENSEMBLE (avg)':16s} test acc = {m['accuracy']*100:.2f}%   macro-F1 = {m['macro_f1']:.4f}")
    print("Ensemble recall:", {c: round(v['recall'], 3) for c, v in m["per_class"].items()})
    return m


def show_predictions(base, ckpt_paths, n=8, seed=0, save_path=None):
    """Same n random TEST images, prediction from every model side by side."""
    import matplotlib.pyplot as plt
    from PIL import Image
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loaded = {name: load_checkpoint(path, device) for name, path in ckpt_paths.items()}
    first_ckpt = next(iter(loaded.values()))[1]
    classes = first_ckpt["classes"]
    random.seed(seed)
    picks = random.sample(first_ckpt["split"]["test"], n)

    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5.6 * rows))
    for ax, p in zip(np.array(axes).flat, picks):
        img = Image.open(os.path.join(base, p)).convert("RGB")
        actual = os.path.basename(os.path.dirname(p))
        ax.imshow(img.convert("L"), cmap="gray"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Actual: {actual}", fontsize=11, fontweight="bold")
        lines = []
        for name, (model, ckpt) in loaded.items():
            _, ev_tf = build_transforms(ckpt["extra"]["img_size"])
            with torch.no_grad():
                pr = torch.softmax(model(ev_tf(img).unsqueeze(0).to(device)), 1)[0]
            pred = classes[pr.argmax().item()]
            lines.append((f"{name}: {pred} ({pr.max()*100:.0f}%)", "green" if pred == actual else "red"))
        for k, (txt, col) in enumerate(lines):
            ax.text(0.5, -0.07 - 0.075 * k, txt, transform=ax.transAxes, ha="center", va="top",
                    color=col, fontsize=10)
    for ax in np.array(axes).flat[len(picks):]:
        ax.axis("off")
    plt.suptitle("Predictions from all models on held-out test images (green = correct, red = wrong)")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()