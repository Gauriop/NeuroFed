"""f1engine.py - confusion matrix, accuracy, precision/recall/F1, plotting.

Everything (including the plot title accuracy) is computed from the confusion
matrix, so the title always matches the diagonal.
"""
import numpy as np
import torch


class F1Engine:
    def __init__(self, class_names):
        self.class_names = list(class_names)
        self.n = len(self.class_names)
        self.reset()

    def reset(self):
        self.cm = np.zeros((self.n, self.n), dtype=np.int64)  # rows = true, cols = predicted

    def update(self, y_true, y_pred):
        y_true = torch.as_tensor(y_true).view(-1).cpu().numpy()
        y_pred = torch.as_tensor(y_pred).view(-1).cpu().numpy()
        np.add.at(self.cm, (y_true, y_pred), 1)

    @torch.no_grad()
    def evaluate(self, model, loader, device="cpu"):
        self.reset()
        model.eval()
        for x, y in loader:
            preds = model(x.to(device)).argmax(1)
            self.update(y, preds)
        return self.metrics()

    def accuracy(self):
        total = self.cm.sum()
        return float(np.trace(self.cm) / total) if total else 0.0

    def metrics(self):
        cm = self.cm.astype(float)
        tp = np.diag(cm)
        precision = np.divide(tp, cm.sum(0), out=np.zeros(self.n), where=cm.sum(0) > 0)
        recall = np.divide(tp, cm.sum(1), out=np.zeros(self.n), where=cm.sum(1) > 0)
        denom = precision + recall
        f1 = np.divide(2 * precision * recall, denom, out=np.zeros(self.n), where=denom > 0)
        support = cm.sum(1)
        return {
            "accuracy": self.accuracy(),
            "macro_f1": float(f1.mean()),
            "weighted_f1": float((f1 * support).sum() / support.sum()) if support.sum() else 0.0,
            "per_class": {
                c: {"precision": float(precision[i]), "recall": float(recall[i]),
                    "f1": float(f1[i]), "support": int(support[i])}
                for i, c in enumerate(self.class_names)
            },
            "confusion_matrix": self.cm.tolist(),
        }

    def plot_confusion_matrix(self, path, arch, split_name="Test"):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(1.2 * self.n + 3, 1.2 * self.n + 2.5))
        ax.imshow(self.cm, cmap="Blues")
        ax.set_xticks(range(self.n)); ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticks(range(self.n)); ax.set_yticklabels(self.class_names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        # Title is built from the real arch + the accuracy computed from THIS matrix.
        ax.set_title(f"{arch} ({split_name} Accuracy: {self.accuracy() * 100:.1f}%)")
        thresh = self.cm.max() / 2 if self.cm.max() else 1
        for i in range(self.n):
            for j in range(self.n):
                ax.text(j, i, int(self.cm[i, j]), ha="center", va="center",
                        color="white" if self.cm[i, j] > thresh else "black")
        fig.tight_layout()
        fig.savefig(path, dpi=200)
        plt.close(fig)