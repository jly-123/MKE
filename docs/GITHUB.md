# Preparing the GitHub repository

The extracted directory is the repository root. Upload its **contents**, so `README.md`, `mke_new.py`, `config/`, and `.github/` are directly at the GitHub repository root.

## Check the source

```bash
python scripts/verify_repository.py --check-manifest
python test_mke_new_group_shapley.py
python -m unittest discover -s tests -v
```

`.gitignore` excludes generated results, caches, environment-specific files and unlisted data/weights. The six published CSVs and six best checkpoints are explicitly allowed. Preserve `LICENSE`, `THIRD_PARTY_NOTICES.md` and the source provenance record.

After creating an empty repository on GitHub, initialize the extracted folder and inspect the staged files:

```bash
git init
git add .
git status --short
git diff --cached --stat
git commit -m "Add MKE source and experiment tools"
git branch -M main
```

Replace the account and repository placeholders before running:

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/MKE.git
git push -u origin main
```

These commands are instructions for publishing; preparing this archive does not publish a remote repository.

## Release information

Add the final author list, manuscript citation and public repository URL when publishing them. Choose a release tag and record the exact commit with experiments. The included data and historical weights have task configurations, split counts and checksum manifests; known data and checkpoint provenance limits are documented in `docs/DATA_VALIDATION.md`.

GitHub Actions checks source structure, mathematical behavior and experiment tools. It does not download data or weights, install the GNN stack, or use a GPU.

After intentional source changes, regenerate the checksum manifest:

```bash
python scripts/verify_repository.py --write-manifest
```
