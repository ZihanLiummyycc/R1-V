# Sports Game Agent
Fine-Tuning Vision Language Model for Video Games


A lightweight pipeline for **data generation → supervised fine-tuning (SFT) → evaluation**.  
The repository organizes the workflow into three stages driven by dedicated scripts.



## Table of Contents
- [Overview](#overview)
- [Scripts](#scripts)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
  - [A) Data Generation](#a-data-generation)
  - [B) Supervised-Fine Tuning (SFT)](#b-supervised-fine-tuning-sft)
  - [C) Evaluation](#c-evaluation)
- [Configuration](#configuration)
- [Outputs & Logging](#outputs--logging)
- [Reproducibility](#reproducibility)
- [Troubleshooting](#troubleshooting)
- [Acknowledgements](#acknowledgements)

---

## Overview

This project contains three components:

- **Data generation:** `call_detetction.py`, `cc_classifier.py`, `get_coord.py`  
- **Supervised fine-tuning (SFT):** `sft_normal.py`  
- **Evaluation:** `atari_play_lrc.py`

Use the data generation scripts to prepare training data, run SFT to train a model, and then evaluate results with the evaluation script.

---

## Repository Structure

```text
.
├── data/                     # Datasets & intermediate artifacts (consider .gitignore large files)
│   ├── raw/                  # Raw/unprocessed inputs
│   ├── processed/            # Cleaned/derived artifacts
│   └── splits/               # Train/val/test splits (optional)
├── outputs/
│   ├── sft/                  # Saved checkpoints, logs
│   └── eval/                 # Evaluation reports, metrics
├── requirements.txt
├── README.md
└── *.py                      # Project scripts (see below)
