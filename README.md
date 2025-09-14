# Sports Game Agent
Fine-Tuning Vision Language Model for Video Games


A lightweight pipeline for **data generation → supervised fine-tuning (SFT) → evaluation**.  
The repository organizes the workflow into three stages driven by dedicated scripts.


## Overview

This project contains three components:

- **Data generation:** `call_detetction.py`, `cc_classifier.py`, `get_coord.py`  
- **Supervised fine-tuning (SFT):** `sft_normal.py`  
- **Evaluation:** `atari_play_lrc.py`

Use the data generation scripts to prepare training data, run SFT to train a model, and then evaluate results with the evaluation script.

---


