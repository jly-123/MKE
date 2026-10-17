import csv
import json
import os
import sys
from datetime import datetime

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch
from torch_geometric.utils import add_remaining_self_loops

from mke_support.data import MoleculeDataset
from mke_support.gnn import get_gnnNets
from mke_support.release import file_sha256, validate_input_csv
from mke_support.utils import compatible_state_dict, configure_paths, resolve_input_csv


PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))


def summarize_predictions(rows):
    if not rows:
        raise ValueError("No test graphs remain after molecular parsing")
    confusion = [[0, 0], [0, 0]]
    label_counts = {"0": 0, "1": 0}
    prediction_counts = {"0": 0, "1": 0}
    for row in rows:
        label = int(row["label"])
        prediction = int(row["prediction"])
        if label not in (0, 1) or prediction not in (0, 1):
            raise ValueError("Predictor evaluation requires binary labels and predictions")
        confusion[label][prediction] += 1
        label_counts[str(label)] += 1
        prediction_counts[str(prediction)] += 1
    num_correct = confusion[0][0] + confusion[1][1]
    return {"accuracy": num_correct / len(rows), "num_correct": num_correct,
            "count": len(rows), "label_counts": label_counts,
            "prediction_counts": prediction_counts, "confusion_matrix": confusion,
            "confusion_matrix_rows": "true label in [0, 1]",
            "confusion_matrix_columns": "predicted label in [0, 1]"}


@hydra.main(config_path="config", config_name="config")
def main(config):
    configure_paths(config, PACKAGE_ROOT)
    dataset_name = str(config.datasets.dataset_name)
    config.models.param = config.models.param[dataset_name]
    config.models.param.add_self_loop = False
    if not config.models.param.graph_classification:
        raise ValueError("This entry evaluates graph classification checkpoints")
    batch_size = config.performance.explain_batch_size
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("performance.explain_batch_size must be a positive integer")
    if torch.cuda.is_available():
        if not 0 <= int(config.device_id) < torch.cuda.device_count():
            raise ValueError("device_id is outside the visible CUDA device range")
        device = torch.device("cuda", index=config.device_id)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("Using GPU: {}".format(torch.cuda.get_device_name(device)))
    else:
        device = torch.device("cpu")
        print("CUDA is unavailable; evaluating on CPU")

    input_csv = resolve_input_csv(config, PACKAGE_ROOT)
    label_column = str(config.datasets.label_column)
    input_report = validate_input_csv(input_csv, label_column)
    checkpoint_path = os.path.join(config.models.gnn_saving_dir, dataset_name,
                                   "{}_{}l_best.pth".format(config.models.gnn_name, len(config.models.param.gnn_latent_dim)))
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("Required checkpoint not found: {}".format(checkpoint_path))
    frame = pd.read_csv(input_csv)
    train_index, test_index, valid_index = [], [], []
    dataset = MoleculeDataset()
    dataset.build_mol_graph_data(dataset_smiles=frame, train_index=train_index,
                                 test_index=test_index, valid_index=valid_index,
                                 labels_name=label_column, group=["training", "valid", "test"])
    dataset.data.x = dataset.data.x.float()
    dataset.data.y = dataset.data.y.squeeze().long()
    if not test_index:
        raise ValueError("No test graphs remain after molecular parsing")
    if dataset.num_classes != 2:
        raise ValueError("Expected both binary classes after molecular parsing")

    model = get_gnnNets(input_dim=dataset.num_node_features, output_dim=dataset.num_classes,
                       model_config=config.models)
    state_dict = compatible_state_dict(torch.load(checkpoint_path, map_location="cpu")["net"])
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    rows = []
    for batch_start in range(0, len(test_index), batch_size):
        indices = test_index[batch_start:batch_start + batch_size]
        graphs = [dataset[index] for index in indices]
        labels = [int(graph.y.item()) for graph in graphs]
        node_counts = [int(graph.num_nodes) for graph in graphs]
        for graph in graphs:
            graph.edge_index = add_remaining_self_loops(graph.edge_index, num_nodes=graph.num_nodes)[0]
        batch = Batch.from_data_list(graphs).to(device)
        with torch.no_grad():
            logits = model(batch.x, batch.edge_index, batch.batch)
            probabilities = torch.softmax(logits, dim=-1).cpu()
        if tuple(probabilities.shape) != (len(indices), 2) or not bool(torch.isfinite(probabilities).all().item()):
            raise ValueError("Predictor produced nonfinite probabilities or an unexpected output shape")
        predictions = probabilities.argmax(-1).tolist()
        values = probabilities.tolist()
        for local_index, sample_index in enumerate(indices):
            rows.append({"sample_index": int(sample_index), "label": labels[local_index],
                         "prediction": int(predictions[local_index]),
                         "p0": values[local_index][0], "p1": values[local_index][1],
                         "num_nodes": node_counts[local_index]})
        print("Evaluated {}/{} test graphs".format(len(rows), len(test_index)), flush=True)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    output_dir = os.path.join(config.explainers.explanation_result_dir, dataset_name,
                              str(config.models.gnn_name), "predictor_evaluation", run_id)
    os.makedirs(output_dir, exist_ok=False)
    summary = summarize_predictions(rows)
    summary.update({"dataset": dataset_name, "run_id": run_id,
                    "input_sha256": input_report["sha256"],
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                    "metric_scope": "all retained test graphs"})
    metadata = {"dataset": dataset_name, "run_id": run_id, "device": str(device),
                "input_csv": os.path.abspath(input_csv), "input_report": input_report,
                "label_column": label_column, "checkpoint": os.path.abspath(checkpoint_path),
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "implementation_sha256": file_sha256(__file__),
                "num_valid_graphs": len(dataset), "num_skipped_rows": input_report["count"] - len(dataset),
                "retained_split_counts": {"training": len(train_index), "valid": len(valid_index), "test": len(test_index)},
                "sample_index_scope": "index in the molecular dataset after parsing, matching mke_new.py",
                "inference": "clean graph, explicit remaining self-loops, model self-loop insertion disabled, no AMP",
                "batch_size": batch_size,
                "software_versions": dict({"python": sys.version}, **{
                    name: getattr(sys.modules.get(name), "__version__", "unknown")
                    for name in ("torch", "torch_geometric", "rdkit", "pandas", "hydra", "omegaconf")})}
    with open(os.path.join(output_dir, "predictions.csv"), "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_index", "label", "prediction", "p0", "p1", "num_nodes"])
        writer.writeheader()
        writer.writerows(rows)
    for filename, value in (("predictor_summary.json", summary), ("predictor_metadata.json", metadata)):
        with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
    with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as stream:
        stream.write(OmegaConf.to_yaml(config))
    print("Accuracy: {}/{} = {:.6f}".format(summary["num_correct"], summary["count"], summary["accuracy"]))
    print("Predictor evaluation saved to: {}".format(output_dir))


if __name__ == "__main__":
    main()
