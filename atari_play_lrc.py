import torch
import numpy as np
import json
import re
from qwen_vl_utils import process_vision_info
from PIL import Image
import os
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import shutil
import gymnasium as gym
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor, DummyVecEnv
from stable_baselines3.common.atari_wrappers import AtariWrapper
import time
from datasets import load_from_disk
import random
import argparse
from peft import PeftModel
from test_utils import SYSTEM_PROMPT, user_prompt, all_env_action_spaces, knowledgebase
# from src.r1-v.src.open_r1.knowedge_base import knowledgebase

def parse_args():
    """
    Parse command line arguments for evaluation script.
    """
    parser = argparse.ArgumentParser(description="Game Agent Evaluation Script")
    parser.add_argument('--index', type=int, default=-1, required=True, help='Index of this test')
    
    parser.add_argument('--ICL', action='store_true', help='Index of this test')
    parser.add_argument('--RAG', action='store_true', help='if rag')
    parser.add_argument('--lora_model_path', type=str, default=None, help='Path to the pretrained LoRA model directory')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the pretrained model directory')
    parser.add_argument('--limits', type=int, default=1000, help='Maximum steps per game episode')
    parser.add_argument('--saved_path', type=str, required=True, help='Directory to save evaluation outputs')
    parser.add_argument('--game_name', type=str, required=True, help='Name of the game to run if index == -1')
    parser.add_argument('--ICL_path', type=str, help='path to icl')
    return parser.parse_args()


def inference(model, processor, example, ICL_examples=None):
    """
    Run inference using the vision-language model to predict the next action based on input images and context.

    Args:
        model: The loaded vision-language model.
        processor: Processor to tokenize and prepare input.
        example: Dictionary containing the current example to be processed.
        ICL_examples: List of in-context learning examples to append.

    Returns:
        The generated reasoning text from the model.
    """
    # Prepare messages including in-context examples
    messages = []
    if ICL_examples is not None:
        for eg in ICL_examples:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": eg["problem"]},
                    # Use last 3 images for context if available
                    *([{"type": "image", "image": img} for img in eg.get("images", [])[-3:]]),
                    {"type": "text", "text": eg["answers"]},
                ]
            })

    # Append the current example at the end
    messages += [
        {"role": "system", "content": [{"type": "text", "text": example["system"]}]},
        {"role": "user", "content": [
            {"type": "text", "text": example["problem"]},
            *([{"type": "image", "image": img} for img in example.get("images", [])[-3:]])
        ]}
    ]

    # Build the prompt string from messages
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # print(prompt)
    # Process the images and prepare input tensor
    input_image, _ = process_vision_info(messages)

    inputs = processor(
        text=[prompt],
        images=input_image,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    # Generate output from the model with sampling parameters

    generated_ids = model.generate(
        **inputs,
        use_cache=True,
        max_new_tokens=3000,
        do_sample=True,
        temperature=0.8,  # Higher temperature for diversity
        top_p=0.9,        # Nucleus sampling probability
        top_k=3,          # Top-k sampling
    )
    # print(generated_ids)
    # Decode the generated tokens to string
    batch_output_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # Extract the assistant's response part
    res = batch_output_text[0].split("\nassistant\n")[-1]
    # print(res)
    return res

def check_file(source_file, target_file):
    """
    Ensure the target file exists by copying from source if missing.
    """
    if not os.path.exists(target_file):
        print(f"{target_file} does not exist. Copying from {source_file}...")
        if os.path.exists(source_file):
            shutil.copy(source_file, target_file)
            print(f"Successfully copied {source_file} to {target_file}")
        else:
            print(f"Error: Source file {source_file} does not exist.")
    else:
        print(f"{target_file} already exists. No need to copy.")
        
    import sys
    sys.stdout.flush()

def load_checkpoint(model_path, lora_model_path=None):
    """
    Load the base or LoRA-finetuned vision-language model and its processor.
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)

    if lora_model_path is not None:
        # Load LoRA fine-tuned model on top of base model
        peft_model = PeftModel.from_pretrained(model, lora_model_path)
    else:
        peft_model = model

    return peft_model, processor

def make_atari_env(env_id, clip_reward=False, render_mode='rgb_array'):
    """
    Create a wrapped Atari environment with optional reward clipping and rendering mode.
    """
    env = gym.make(env_id, render_mode=render_mode)
    env = AtariWrapper(env, clip_reward=clip_reward)
    return env

def main():
    args = parse_args()

    model_path = args.model_path
    limits = args.limits
    saved_path = args.saved_path
    os.makedirs(saved_path, exist_ok=True)

    if args.index == -1:
        # Run only the specific game if index == -1
        game_names = [args.game_name]
    if not game_names:
        raise ValueError(f"Invalid index {args.index}, no games found.")

    print(f"Model path: {model_path}", flush=True)
    print(f"Saved path: {saved_path}", flush= True)
    
    if args.ICL_path is not None:
        args.ICL = True
        print(f"Using in-context learning examples from: {args.ICL_path}, the flag is {args.ICL}")

    # Ensure config files needed by processor exist in model path
    check_file("/data/user/qxiao183/qxiao183test2/GameAgent/GA_branches/resources/model/preprocessor_config.json", os.path.join(model_path, "preprocessor_config.json"))
    check_file("/data/user/qxiao183/qxiao183test2/GameAgent/GA_branches/resources/model/chat_template.json", os.path.join(model_path, "chat_template.json"))

    # Load the model and processor
    model, processor = load_checkpoint(model_path, args.lora_model_path)

    replay_info = {}
    data = {}

    for game_name in game_names:
        replay_info[game_name] = {"images": [], "reasoning": []}

        # Prepare directories for saving outputs
        game_saved_path = os.path.join(saved_path, game_name)
        os.makedirs(game_saved_path, exist_ok=True)
        game_image_saved_path = os.path.join(game_saved_path, "images")
        os.makedirs(game_image_saved_path, exist_ok=True)

        # Derive short game name for mapping actions
        short_game_name = game_name.split("NoFrameskip-v4")[0]
        print(f"Starting evaluation for game: {game_name} (short name: {short_game_name})")

        # Get action mapping dict for this game
        def get_action_meanings(game_name):
            env = gym.make(f"{game_name}NoFrameskip-v4")
            meanings = env.unwrapped.get_action_meanings()
            return {i: name for i, name in enumerate(meanings)}
        action_to_text = get_action_meanings(short_game_name)
        if action_to_text is None:
            raise ValueError(f"Action mapping for game {short_game_name} not found in {all_env_action_spaces.keys()}.")
        text_to_action = {v: k for k, v in action_to_text.items()}

        # Load RL policy checkpoint for action prediction baseline
        env = DummyVecEnv([lambda: make_atari_env(game_name, clip_reward=False)])
        env = VecFrameStack(env, n_stack=4)

        if args.ICL:
            # Load in-context learning dataset for current game
            ICL_pool = load_from_disk(args.ICL_path)["train"]

        gifs = []
        reasoning_generated = []
        ep = 0
        step = 0
        ep_return = 0  # total reward accumulator

        while True:
            # Reset environment for each episode
            env = DummyVecEnv([lambda: make_atari_env(game_name, clip_reward=False)])
            env = VecFrameStack(env, n_stack=4)
            obs = env.reset()
            done = False

            # Initialize image buffer with the first rendered frame repeated 3 times for context
            img_np = env.render(mode='rgb_array')
            img_pil = Image.fromarray(img_np)
            image_buffer = [img_pil] * 3
            gifs.append(img_pil)

            while not done and step <= limits:
                # Sample 3 random in-context learning examples for prompting
                if args.ICL:
                    ICL_examples = random.sample(list(ICL_pool), 3)
                else:
                    ICL_examples = None

                if args.RAG:
                    knowledge = knowledgebase[short_game_name]
                    final_user_prompt = user_prompt + knowledge
                else:
                    final_user_prompt = user_prompt
                # Construct current example for inference
                example = {
                    "system": SYSTEM_PROMPT,
                    "problem": final_user_prompt,
                    "images": image_buffer
                }

                # Render current frame and save image to disk
                img_np = env.render(mode='rgb_array')
                img_pil = Image.fromarray(img_np)
                image_path = os.path.join(game_image_saved_path, f"{step}.png")
                img_pil.save(image_path)

                # Run model inference to get reasoning text and predicted action
                reasoning_VLM = inference(model, processor, example, ICL_examples)
                replay_info[game_name]["images"].append(image_path)
                replay_info[game_name]["reasoning"].append(reasoning_VLM)
                reasoning_generated.append(reasoning_VLM)

                # Parse the predicted action from reasoning_VLM text
                if "noR" in model_path:
                    # If model_path includes 'noR', use raw reasoning as action
                    action_VLM = reasoning_VLM.strip()
                else:
                    # Try to extract <answer>...</answer> tag content
                    pattern = r'<answer>(.*?)</answer>'
                    matches = re.findall(pattern, reasoning_VLM)
                    if matches:
                        try:
                            action_VLM = matches[0].strip().split()[-1].replace('.', '')
                        except:
                            print(f"Error parsing action from reasoning: {matches}")
                            action_VLM = "NOOP"
                    else:
                        print(f"Warning: no <answer> tag found in reasoning: {reasoning_VLM}")
                        action_VLM = "NOOP"

                # Convert predicted action text to action index
                try:
                    action_VLM_idx = text_to_action[action_VLM]
                except KeyError as e:
                    print(f"Invalid predicted action '{action_VLM}', defaulting to NOOP. Error: {e}")
                    action_VLM_idx = 0  # NOOP

                # Take a step in environment using predicted action
                for _ in range(4):
                    obs, reward, done, info = env.step([action_VLM_idx])
                    ep_return += reward[0]
                step += 1

                # Update image buffer for next step (sliding window of last 3 images)
                img_np = env.render(mode='rgb_array')
                img_pil = Image.fromarray(img_np)
                image_buffer = image_buffer[1:] + [img_pil]
                gifs.append(img_pil)

                # Periodic logging and save intermediate results
                if step % 50 == 0:
                    print({
                        "reward": ep_return,
                        "episode": ep,
                        "step": step,
                    })
                    data[game_name] = {
                        "reward": ep_return,
                        "eps": ep,
                        "limits": limits,
                        "steps": step,
                    }
                    with open(os.path.join(game_saved_path, "score.json"), "w", encoding="utf-8") as f:
                        json.dump(data[game_name], f, indent=4, ensure_ascii=False)

                if step > limits:
                    # Stop episode if step limit reached
                    break

            ep += 1
            if step > limits:
                break

        # Save final results for this game
        data[game_name] = {
            "reward": ep_return,
            "eps": ep,
            "limits": limits,
            "steps": step,
        }
        print(f"Finished {game_name}, final score: {data[game_name]}")

        # Save replay info and scores to JSON
        with open(os.path.join(game_saved_path, "replay.json"), "w", encoding="utf-8") as f:
            json.dump(replay_info[game_name], f, indent=4, ensure_ascii=False)
        with open(os.path.join(game_saved_path, "score.json"), "w", encoding="utf-8") as f:
            json.dump(data[game_name], f, indent=4, ensure_ascii=False)
        print(f"Saved JSON files for {game_name}.")

    # Save overall replay and score info across all games
    with open(os.path.join(saved_path, "replay.json"), "w", encoding="utf-8") as f:
        json.dump(replay_info, f, indent=4, ensure_ascii=False)
    with open(os.path.join(saved_path, "score.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()
