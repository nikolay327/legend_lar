# legend_lar

Research code for conditional neural ratio estimation on sparse LAr/HPGe-style detector observables.

The package implements a contrastive neural ratio estimation pipeline for structured detector data represented as variable-length sequences of detector hits and auxiliary event-level features. The code is written in PyTorch and is organized around multimodal encoders, k-fold ensemble training, deep-supervision variants, and empirical calibration/inference utilities.

This repository contains code only. It does not include collaboration data, detector metadata, processed datasets, trained model checkpoints, or experiment-specific configuration files.

## Scope

The main components are:

* multimodal LAr and HPGe encoders for sparse detector observables;
* geometry-aware tokenization of detector-channel coordinates;
* transformer blocks for packed variable-length sequences;
* contrastive neural ratio estimation losses;
* k-fold and bootstrap-based ensemble training;
* HPGe-prefix deep supervision;
* ensemble-based inference and calibration utilities.

The package is research code and is not a standalone reproducible analysis. External data products and configuration files are required to run the full training and inference workflow.

## Method overview

The model encodes LAr-side observables and HPGe-side conditioning information into a shared embedding space. Contrastive objectives are then used to train scores from matched and mismatched LAr/HPGe combinations.

The intended use is to construct conditional neural scores for inference and calibration, rather than a generic binary classifier output. The downstream calibration code uses ensemble predictions and empirical null samples to construct event-level and global p-value-like quantities.

## Deep supervision

The deep-supervision variant uses intermediate HPGe prefixes during training. This allows the model to learn how the LAr-side score changes as additional HPGe-side information becomes available.

The implementation supports:

* prefix-wise contrastive losses;
* configurable prefix weights;
* an interaction auxiliary loss;
* logging of prefix-level training and validation losses.

This part of the code is implemented mainly in `legend_lar.kfold_ensemble.nre_c_ds`.

## Package structure

```text
src/legend_lar/
├── model/              # Encoders, tokenizers, transformer blocks
├── data/               # Iterable datasets and collate functions
├── kfold_ensemble/     # Training code for k-fold/bootstrap ensembles
├── calibration/        # Ensemble inference and empirical calibration
└── utils/              # Configuration, file handling, RNG utilities
```

## Installation

```bash
pip install -e .
```

The code assumes a CUDA-capable PyTorch environment for the GPU-oriented model components. Some components depend on FlashAttention and are intended for GPU/HPC execution.

## Data and configuration

The repository does not contain the data or configuration files used in the original analysis. The training and inference entry points expect externally provided directories containing processed sparse arrays, HPGe feature arrays, detector-coordinate files, model configuration JSON files, and checkpoint/output locations.

## Status

This repository is maintained as research code for method development. Interfaces and configuration formats may change.
