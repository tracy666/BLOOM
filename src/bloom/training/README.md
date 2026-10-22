# General training implementation

BLOOM and the 22 forecasting baselines share one training implementation. Select the model through `python -m bloom.training`; no version-specific training file is required.

| File | Responsibility |
| --- | --- |
| `trainer.py` | Shared input preparation, data loaders, optimization, validation, checkpoint saving and reporting; conditional handling of the two developmental directions |
| `arguments.py` | One advanced argument schema with defaults for baseline, unidirectional BLOOM and bidirectional BLOOM |
| `cli.py` | Public model selection, dataset paths, recipe overrides and launch records |
| `recipes.json` | Installed release recipes and dataset annotation mappings |
| `__main__.py` | Package command entry point |
| `__init__.py` | Package description |

The mathematical models remain in `../models/bloom.py` and `../models/transformer.py`. The trainer chooses the appropriate model rather than duplicating its definition.

## Run from the repository root

```bash
# Inspect the complete resolved settings without loading data.
python -m bloom.training --model bloom-unidirectional --dataset cortex --dry-run

# Inspect the common advanced arguments with the selected model's defaults.
python -m bloom.training --model bloom-bidirectional --native-help
python -m bloom.training --model PatchTST --native-help

# Train with the existing author-panel launchers.
DATASET=cortex bash scripts/train/train_bloom_unidirectional.sh
DATASET=cortex bash scripts/train/train_bloom_bidirectional.sh
DATASET=cortex bash scripts/train/train_PatchTST.sh
```

All 24 existing training launchers keep the same names and public arguments. `DATASET` accepts `prenatal`, `cortex` and `sted`; `PANEL=author` is the default, while `PANEL=hvg` selects the prepared model-HVG panel. Dataset paths, resources and additional examples are described in the repository README and `configs/training/README.md`.

Unidirectional BLOOM uses a separate checkpoint for each trained direction. To train the reverse model with the same launcher:

```bash
DATASET=cortex FM_TEMPORAL_DIRECTION=reverse \
  bash scripts/train/train_bloom_unidirectional.sh
```

Bidirectional BLOOM uses one shared model with direction conditioning. The general trainer retains its directional sampling and biological-stage split rules. Baseline defaults retain the adopted forecasting configuration; they are not replaced by BLOOM's model settings.

## Update an existing checkout

Replace the entire existing `src/bloom/training/` directory with this `training/` directory. Do not merge the folders, because merging would leave the obsolete version-named files in place. The package command and training launchers then continue to use the same public interface.

Use the accompanying complete repository update when running checkpoint export and evaluation: its adapters have also been updated to initialize the general trainer. Replacing only this folder updates training, but leaves older external adapters referring to the removed implementation files.

Historical version labels may remain in checkpoint or prediction-file metadata to identify existing saved formats. They are compatibility fields, not parallel trainer implementations or model choices.
