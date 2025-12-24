#!/usr/bin/env python3
"""
Extract activations from search tool episodes for cross-tool transfer test.

Usage: python scripts/extract_search_activations.py
"""
import sys
sys.path.insert(0, '.')

import json
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

def main():
    # Check for search episodes
    search_file = Path("data/raw/search_episodes_20251221_141706.jsonl")

    if not search_file.exists():
        print(f"❌ Search episodes not found: {search_file}")
        print("Cannot extract search activations without search episodes.")
        return

    print(f"✓ Found search episodes: {search_file}")

    # Load episodes
    with open(search_file) as f:
        episodes = [json.loads(line) for line in f]

    print(f"✓ Loaded {len(episodes)} search episodes")

    # Load model for activation extraction
    print("\nLoading model for activation extraction...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print(f"✓ Model loaded on: {next(model.parameters()).device}")

    # Extract activations
    print(f"\nExtracting activations from {len(episodes)} episodes...")

    activations = []
    tool_used_list = []
    claims_action_list = []
    categories_list = []

    TARGET_LAYER = 16  # Middle layer

    for i, ep in enumerate(tqdm(episodes, desc="Extracting")):
        try:
            # Build full text
            system_prompt = ep.get('system_prompt', '')
            user_turns = ep.get('user_turns', [])
            reply = ep.get('reply', '')

            # Format as conversation
            full_text = f"{system_prompt}\n\n"
            for turn in user_turns:
                full_text += f"User: {turn}\n\n"
            full_text += f"Assistant: {reply}"

            # Tokenize
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Extract activations
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Get activation from target layer at last token
            # hidden_states[0] = embeddings, hidden_states[1] = layer 0, ..., hidden_states[17] = layer 16
            hidden_state = outputs.hidden_states[TARGET_LAYER + 1]  # +1 because index 0 is embeddings

            # Use last token position (before tool call if present)
            activation = hidden_state[0, -1, :].cpu().numpy()

            activations.append(activation)
            tool_used_list.append(ep.get('tool_used', False))
            claims_action_list.append(ep.get('claims_action', False))
            categories_list.append(ep.get('category', 'unknown'))

        except Exception as e:
            print(f"\n⚠️  Failed on episode {i}: {e}")
            continue

    # Convert to arrays
    activations_arr = np.stack(activations)
    tool_used_arr = np.array(tool_used_list, dtype=bool)
    claims_action_arr = np.array(claims_action_list, dtype=bool)
    categories_arr = np.array(categories_list)

    print(f"\n✓ Extracted {len(activations_arr)} activations")
    print(f"  Shape: {activations_arr.shape}")
    print(f"  Tool used rate: {tool_used_arr.mean():.1%}")
    print(f"  Claims action rate: {claims_action_arr.mean():.1%}")

    # Save
    output_file = Path("data/labeled/activations_search.npz")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_file,
        activations=activations_arr,
        tool_used=tool_used_arr,
        claims_action=claims_action_arr,
        categories=categories_arr,
    )

    print(f"\n✓ Saved to: {output_file}")
    print(f"\nCategory distribution:")
    from collections import Counter
    for cat, count in Counter(categories_arr).most_common():
        print(f"  {cat}: {count}")

if __name__ == "__main__":
    main()
