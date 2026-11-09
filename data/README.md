# Input data

Six historical experiment CSVs are included in `data/origin_data/`. Their original bytes, row order and split assignments are preserved:

```text
data/origin_data/
  Mutagenicity.csv
  BBBP.csv
  hERG.csv
  Tox21_AhR.csv
  Bace.csv
  clintox.csv
```

Only files for the datasets you intend to run are required. Each CSV needs `smiles`, the dataset's binary label column, and `group` values `training`, `valid`, and `test`. Use the intended preprocessed files and their stored splits; downloading a similarly named dataset does not guarantee the same molecules, endpoints, or splits.

See [the complete dataset guide](../docs/DATASETS.md) for label mappings, observed local file counts, graph features, validation rules, and ClinTox endpoint selection.

To keep data elsewhere:

```bash
python mke_new.py datasets=Mutagenicity datasets.dataset_root=/absolute/path/csv_directory device_id=0
```

An explicit `input_csv=/absolute/path/file.csv` also works. Generated graph caches are written to `data/processed/`. Do not commit generated caches or unintended data to GitHub; follow the data provider's redistribution terms for any files you choose to publish.

## Published input validation

See [DATA_VALIDATION.md](../docs/DATA_VALIDATION.md) before interpreting results. `input_manifest.json` records file SHA-256, labels, counts, split overlaps and conflicting labels. The published files retain their historical splits and known duplicates.
