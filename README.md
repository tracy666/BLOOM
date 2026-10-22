<div align="center">

<img src="assets/bloom_icon.jpg" width="150" alt="BLOOM flower icon">

# BLOOM

### Biologically Informed Flow Matching for Transcriptomic Virtual Embryo Modeling

Conditional generation of developmental cell populations from ordered expression histories and biological priors.

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![PyTorch](https://img.shields.io/badge/PyTorch-models-EE4C2C?logo=pytorch&logoColor=white)](src/bloom/models/bloom.py)
[![Models](https://img.shields.io/badge/Training-24%20launchers-8877C8)](scripts/train)
[![Checkpoints](https://img.shields.io/badge/Checkpoints-pending%20upload-E6A23C)](checkpoints/README.md)

[Overview](#overview) · [Installation](#installation) · [Data preparation](#data-preparation) · [Training](#training) · [Evaluation](#evaluation-in-manuscript-order) · [Figures](#figure-generation) · [Checkpoints](#checkpoints-and-reproducibility)

</div>

---

## Overview

BLOOM models developmental transcriptomic populations at requested stages. It combines temporal population context, inferred cell correspondences, and developmental, anatomical, programme and regulatory knowledge in a conditional flow-matching model.

The two BLOOM configurations live together in **[`src/bloom/models/bloom.py`](src/bloom/models/bloom.py)**:

- **Unidirectional BLOOM:** one trained developmental direction per checkpoint; forward and reverse models can be trained with the same launcher.
- **Bidirectional BLOOM:** a shared model with developmental-direction conditioning.

The paper's 22 forecasting baselines use one shared factory and conditioning interface in **[`src/bloom/models/transformer.py`](src/bloom/models/transformer.py)**. Their original architectures are obtained through a separate, documented Time-Series-Library [THUML/Time-Series-Library](https://github.com/thuml/Time-Series-Library).
