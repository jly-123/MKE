import csv
import hashlib
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path


DATASET_NAMES = ("Mutagenicity", "bbbp", "hERG", "Tox21_AhR", "Bace", "clintox")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(str(path), "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_catalog(root):
    path = Path(root) / "config" / "dataset_catalog.json"
    with path.open("r", encoding="utf-8") as stream:
        catalog = json.load(stream)
    if isinstance(catalog, dict) and "datasets" in catalog:
        catalog = catalog["datasets"]
    if not isinstance(catalog, dict) or set(catalog) != set(DATASET_NAMES):
        raise ValueError("Dataset catalog must contain exactly: {}".format(", ".join(DATASET_NAMES)))
    for name, entry in catalog.items():
        if not isinstance(entry, dict):
            raise ValueError("Invalid catalog entry for {}".format(name))
        for key in ("csv_filename", "label_column"):
            if not isinstance(entry.get(key), str) or not entry[key].strip():
                raise ValueError("Missing {} for {}".format(key, name))
        if Path(entry["csv_filename"]).name != entry["csv_filename"]:
            raise ValueError("csv_filename must be a filename for {}".format(name))
    return catalog


def validate_input_csv(path, label_column, require_splits=("training", "valid", "test")):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("Required CSV not found: {}".format(path))
    if not isinstance(label_column, str) or not label_column:
        raise ValueError("A nonempty label column name is required")
    allowed_splits = ("training", "valid", "test")
    if any(group not in allowed_splits for group in require_splits):
        raise ValueError("Unknown required split")
    groups = Counter({group: 0 for group in allowed_splits})
    classes = Counter({"0": 0, "1": 0})
    count = 0
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            headers = next(reader)
        except StopIteration:
            raise ValueError("CSV is empty: {}".format(path))
        if any(not header.strip() for header in headers):
            raise ValueError("CSV contains an empty column name")
        duplicates = sorted(header for header, number in Counter(headers).items() if number > 1)
        if duplicates:
            raise ValueError("Duplicate CSV columns: {}".format(", ".join(duplicates)))
        missing = sorted({"smiles", "group", label_column}.difference(headers))
        if missing:
            raise ValueError("Missing CSV columns: {}".format(", ".join(missing)))
        smiles_index = headers.index("smiles")
        group_index = headers.index("group")
        label_index = headers.index(label_column)
        for row in reader:
            line = reader.line_num
            if len(row) != len(headers):
                raise ValueError("CSV line {} has {} fields; expected {}".format(line, len(row), len(headers)))
            if not row[smiles_index].strip():
                raise ValueError("CSV line {} has an empty SMILES value".format(line))
            group = row[group_index]
            if group not in allowed_splits:
                raise ValueError("CSV line {} has invalid group {!r}; use training, valid, or test exactly".format(line, group))
            label_text = row[label_index].strip()
            try:
                label = Decimal(label_text)
            except InvalidOperation:
                raise ValueError("CSV line {} has invalid binary label {!r}".format(line, label_text))
            if not label.is_finite() or label not in (Decimal(0), Decimal(1)):
                raise ValueError("CSV line {} has invalid binary label {!r}".format(line, label_text))
            classes[str(int(label))] += 1
            groups[group] += 1
            count += 1
    if not count:
        raise ValueError("CSV contains no data rows")
    missing_splits = [group for group in require_splits if groups[group] == 0]
    if missing_splits:
        raise ValueError("CSV has no rows for required splits: {}".format(", ".join(missing_splits)))
    if any(classes[label] == 0 for label in ("0", "1")):
        raise ValueError("CSV must contain both binary classes 0 and 1")
    return {"path": str(path), "sha256": file_sha256(path), "count": count,
            "groups": dict(groups), "classes": dict(classes), "label_column": label_column}
