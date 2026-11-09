# WMS Relation-GAT + Immune Memory Online 3D Packing

Research-only synthetic mixed-loading rules; **not legal/safety compliance certification**.

## Contents
- `data/raw`: uploaded RS/CSD benchmarks (unchanged)
- `data/processed`: synthetic-attribute V3 datasets
- `src/data_utils_v3.py`: deterministic augmentation and relation rules
- `scripts/create_splits.py`: reproducible 70/15/15 sequence-level split
- `reports`: original validation summaries
- `notebooks/00_colab_setup.ipynb`: GPU setup, dataset checks, split generation

## GitHub + Colab setup
1. Create an **empty private** GitHub repository named `wms-gat-immune-packing`.
2. On your local computer with Git and Git LFS installed, unzip this project and run:
   ```bash
   git lfs install
   git init
   git add .
   git commit -m "Initialize benchmark data and V3 rules"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/wms-gat-immune-packing.git
   git push -u origin main
   ```
3. In Colab, select **Runtime > Change runtime type > GPU**, then run the notebook.
4. Git LFS may have quota limits; if pushing large data is blocked, store datasets in Google Drive and track only SHA256 manifests in Git.

## Reproduce
```bash
python -m scripts.create_splits --seed 42
python -m pytest -q
```

## Research caveats
- V3 synthetic labels reflect authored rules, not actual IMDG/HACCP compliance.
- Dataset dimensional units and item weights are not independently verified; `weight_sim` is simulated.
- Sequence-level split does not ensure unseen attribute combinations; add OOD tests before model evaluation.
- Before training, update rules to the agreed 3-channel box-level model (odor, contamination, physical damage) and move temperature to hard constraints.
