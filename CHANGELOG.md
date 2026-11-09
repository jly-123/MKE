# Changelog

## 2026.11.09

- Restore all support, configuration, script, documentation, test and CI directories omitted from the initial upload.
- Include six original experiment CSVs and six historical best GCN checkpoints with SHA-256 manifests.
- Document exact-SMILES split overlaps, AhR label conflicts and historical checkpoint provenance limits.
- Normalize source text to LF while preserving original CSV and checkpoint bytes.

## 2026.10.18

- Standalone MKE entry with merged-player Shapley group values and greedy selection under a distinct-node budget.
- Molecular GCN, candidate generation, graph scoring, optional predictor training, and dataset-specific configurations for six binary tasks.
- Sequential experiment runner, input and environment checks, raw-result aggregation, statistical summaries, metric figures, and molecular explanation rendering.
- Input/checkpoint hashes, effective run configuration, per-graph results, and greedy-decision traces.
- English/Chinese usage guides, method/configuration/data/output documentation, source provenance, GPLv3 license, and GitHub test workflow.

Default Mutagenicity scoring and selection settings are retained. New entry wiring exposes dataset selection and runtime settings; the group-value core, candidate extraction, scorer, and predictor computations are unchanged from the source snapshot used to prepare the release.
