# GCN checkpoints

Six historical best GCN checkpoints are included at the paths below. File hashes are recorded in `checkpoint_manifest.json`. These files have not been loaded or evaluated during this publication update; historical task provenance is not established by their filenames. For a new experiment, train a predictor with explicit task and input records.

```text
checkpoints/
  Mutagenicity/gcn_3l_best.pth
  bbbp/gcn_3l_best.pth
  hERG/gcn_3l_best.pth
  Tox21_AhR/gcn_3l_best.pth
  Bace/gcn_3l_best.pth
  clintox/gcn_3l_best.pth
```

The `.pth` file must contain a `net` state dictionary compatible with the selected GCN configuration. BBBP uses three hidden layers of width 128; the other configurations use width 256. All use nine input features, two output classes, and global max pooling. Loading is strict, with the supplied compatibility helper handling the legacy GCN parameter naming convention. See [configuration](../docs/CONFIGURATION.md) for the full settings.

The checkpoint must match the input representation and label endpoint. In particular, `clintox` defaults to `CT_TOX`; a two-class output shape alone cannot distinguish it from `FDA_APPROVED`. See [the endpoint guidance](../docs/DATASETS.md#clintox-endpoint).

Use checkpoints from a trusted source, since PyTorch checkpoint loading can deserialize Python objects. The inference entry records each checkpoint's SHA-256 in its run metadata.

To use a separate checkpoint root:

```bash
python mke_new.py datasets=bbbp models.gnn_saving_dir=/absolute/path/checkpoints device_id=0
```

To train a new predictor with the supplied CSV splits:

```bash
python train_gcn.py datasets=bbbp models.gnn_saving_dir=checkpoints_new device_id=0
python mke_new.py datasets=bbbp models.gnn_saving_dir=checkpoints_new device_id=0
```

The trainer writes best and latest checkpoints plus `training_config.yaml` and `training_inputs.json`. The best model is selected by validation accuracy. Existing best/latest checkpoint files are protected by default; use a separate output root for a new run. Setting `overwrite_checkpoint=true` explicitly permits replacing those files.

A newly trained model defines a new predictor run. Retain its training records and use the same selected checkpoint for any explanations you wish to compare.

Publication validation and the ClinTox endpoint limitation are described in [DATA_VALIDATION.md](../docs/DATA_VALIDATION.md). CSV and checkpoint hashes identify supplied files, but do not establish which input bytes or label endpoint were used during historical training.
