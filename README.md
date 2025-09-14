# Sports Game Agent  
Fine-Tuning Vision Language Model for Video Games  

A lightweight pipeline for **data generation → supervised fine-tuning (SFT) → evaluation**.  
The repository organizes the workflow into three stages, each driven by dedicated scripts.  

---

## Overview  

This project contains three components:  

1. **Data generation**  
   - `call_detection.py`  
   - `cc_classifier.py`  
   - `get_coord.py`  

2. **Supervised fine-tuning (SFT)**  
   - `sft_normal.py`  

3. **Evaluation**  
   - `atari_play_lrc.py`  

---

## Environment Setup  

Make sure you have the required dependencies installed:  

bash
pip install -r requirements.txt
Alternatively, run the setup script:

bash
Copy code
bash setup.sh
Data Generation
Step 1: Collect Test Images for an Atari Game
bash
Copy code
python call_detection.py
Step 2: Write a Config File for the Game
See the example config file configs/bowling.yaml for the Bowling game.
You may modify it to suit your game.

You can interactively get the coordinate and color of any pixel from a game image:

bash
Copy code
python get_coord.py --img <image_path>
Step 3: Run the Classifier
bash
Copy code
python cc_classifier.py --input_dir <input_directory> --config_path <config_file_path>
Example for Bowling:

bash
Copy code
python cc_classifier.py --input_dir images/Bowling --config_path configs/bowling.yaml
Supervised Fine-Tuning (SFT)
We provide five different training parameter sets, each defined in a separate script:

small_sft.sh

middle_sft.sh

full_sft.sh

withthink_sft.sh

nothink_sft.sh

Run training using any of the above scripts. For example:

bash
Copy code
bash small_sft.sh
The training state will be saved into one of the following JSON files depending on the script used:

sft_sports_5000_training_state.json

sft_sports_small_training_state.json

sft_sports_full_training_state.json

sft_sports_withthink_training_state.json

sft_sports_nothink_training_state.json

Evaluation
After training, evaluate the fine-tuned model with:

bash
Copy code
python atari_play_lrc.py
Repository Structure
plaintext
Copy code
.
├── configs/                     # Game-specific config files
│   └── bowling.yaml
├── images/                      # Collected game images
│   └── Bowling/
├── call_detection.py            # Step 1: Collect test images
├── get_coord.py                 # Step 2: Get pixel coordinate & color
├── cc_classifier.py             # Step 3: Run classifier
├── sft_normal.py                # Core SFT training script
├── small_sft.sh                 # SFT (small config)
├── middle_sft.sh                # SFT (medium config)
├── full_sft.sh                  # SFT (full config)
├── withthink_sft.sh             # SFT (with think)
├── nothink_sft.sh               # SFT (no think)
├── sft_sports_5000_training_state.json
├── sft_sports_small_training_state.json
├── sft_sports_full_training_state.json
├── sft_sports_withthink_training_state.json
├── sft_sports_nothink_training_state.json
├── atari_play_lrc.py            # Evaluation script
├── requirements.txt             # Dependencies
├── setup.sh                     # Environment setup script
└── README.md
