import numpy as np
from torch.utils.data import Subset


def split_iid(dataset, num_clients=5, seed=42):
    np.random.seed(seed)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    client_indices = np.array_split(indices, num_clients)
    client_datasets = [Subset(dataset, idx.tolist()) for idx in client_indices]
    return client_datasets


def split_non_iid(dataset, num_clients=5, classes_per_client=1, seed=42):
    np.random.seed(seed)

    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([s[1] for s in dataset.samples])

    num_classes = len(np.unique(labels))
    class_indices = {c: np.where(labels == c)[0].tolist() for c in range(num_classes)}
    for c in class_indices:
        np.random.shuffle(class_indices[c])

    client_main_classes = [
        [(i * classes_per_client + j) % num_classes for j in range(classes_per_client)]
        for i in range(num_clients)
    ]

    client_idx_lists = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        owners = [i for i, mc in enumerate(client_main_classes) if c in mc]
        pool = class_indices[c]

        if owners:
            main_cut = int(0.85 * len(pool))
            main_part = pool[:main_cut]
            leftover = pool[main_cut:]

            splits = np.array_split(main_part, len(owners))
            for owner, split in zip(owners, splits):
                client_idx_lists[owner].extend(split.tolist())

            for idx in leftover:
                client_idx_lists[np.random.randint(num_clients)].append(idx)
        else:
            for idx in pool:
                client_idx_lists[np.random.randint(num_clients)].append(idx)

    client_datasets = [Subset(dataset, idx) for idx in client_idx_lists]
    return client_datasets


def summarize_split(client_datasets, dataset, split_name=""):
    if hasattr(dataset, "targets"):
        labels = np.array(dataset.targets)
    else:
        labels = np.array([s[1] for s in dataset.samples])

    print(f"\n=== {split_name} split summary ===")
    for i, cds in enumerate(client_datasets):
        client_labels = labels[cds.indices]
        counts = np.bincount(client_labels, minlength=labels.max() + 1)
        print(f"Client {i+1}: total={len(cds)} | class counts={counts.tolist()}")