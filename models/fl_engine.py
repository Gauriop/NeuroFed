import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def train_local_model(model, dataset, device, epochs=2, lr=0.001, batch_size=32):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


def default_aggregate_fn(global_weights, client_weights_list, client_sizes):
    total_size = sum(client_sizes)
    new_weights = copy.deepcopy(client_weights_list[0])

    for key in new_weights.keys():
        new_weights[key] = torch.zeros_like(new_weights[key], dtype=torch.float32)
        for weights, size in zip(client_weights_list, client_sizes):
            new_weights[key] += weights[key].float() * (size / total_size)

    return new_weights


def default_client_update_fn(model):
    return {k: v.clone() for k, v in model.state_dict().items()}


def evaluate_model(model, test_dataset, device, batch_size=32):
    loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="macro", zero_division=0
    )
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def run_federated_training(
    base_model_fn,
    client_datasets,
    test_dataset,
    device,
    num_rounds=15,
    local_epochs=2,
    aggregate_fn=default_aggregate_fn,
    client_update_fn=default_client_update_fn,
    client_selection_fn=None,
    verbose=True,
):
    global_model = base_model_fn()
    global_model.to(device)
    history = []

    for rnd in range(num_rounds):
        if client_selection_fn:
            participating = client_selection_fn(client_datasets, rnd)
        else:
            participating = client_datasets

        client_weights_list = []
        client_sizes = []

        global_weights = {k: v.clone() for k, v in global_model.state_dict().items()}

        for client_data in participating:
            local_model = base_model_fn()
            local_model.load_state_dict(global_weights)
            local_model = train_local_model(local_model, client_data, device, epochs=local_epochs)

            update = client_update_fn(local_model)
            client_weights_list.append(update)
            client_sizes.append(len(client_data))

        new_global_weights = aggregate_fn(global_weights, client_weights_list, client_sizes)
        global_model.load_state_dict(new_global_weights)

        metrics = evaluate_model(global_model, test_dataset, device)
        metrics["round"] = rnd + 1
        history.append(metrics)

        if verbose:
            print(f"Round {rnd+1}/{num_rounds} | acc={metrics['accuracy']:.4f} | f1={metrics['f1']:.4f}")

    return global_model, history