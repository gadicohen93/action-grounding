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

    # Helper functions for position finding
    def find_token_positions(token_ids, tokenizer):
        """Find key token positions in the sequence."""
        positions = {
            "first_assistant": None,
            "before_tool": None,
            "tool_start": None,
        }

        # Convert to list
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()

        # Find "<<CALL" tokens
        call_str = "<<CALL"
        call_tokens = tokenizer.encode(call_str, add_special_tokens=False)

        # Search for first occurrence of call sequence
        for i in range(len(token_ids) - len(call_tokens) + 1):
            if token_ids[i:i+len(call_tokens)] == call_tokens:
                positions["tool_start"] = i
                positions["before_tool"] = max(0, i - 1)
                break

        # Find first assistant token (after last user message)
        # Look for patterns like "[/INST]" or "Assistant:"
        for marker in ["[/INST]", "Assistant:"]:
            marker_tokens = tokenizer.encode(marker, add_special_tokens=False)
            for i in range(len(token_ids) - len(marker_tokens) + 1):
                if token_ids[i:i+len(marker_tokens)] == marker_tokens:
                    positions["first_assistant"] = i + len(marker_tokens)
                    break
            if positions["first_assistant"] is not None:
                break

        return positions

    def get_safe_probe_index(positions, preference="before_tool"):
        """Get a safe position that's before any tool syntax."""
        if preference == "before_tool" and positions["before_tool"] is not None:
            return positions["before_tool"]
        elif positions["first_assistant"] is not None:
            return positions["first_assistant"]
        else:
            # Fallback: use middle of sequence
            return None

    # Extract activations
    print(f"\nExtracting activations from {len(episodes)} episodes...")
    print("Using 'before_tool' position (same as original analysis)")

    activations = []
    tool_used_list = []
    claims_action_list = []
    categories_list = []
    skipped = 0

    TARGET_LAYER = 16  # Middle layer

    for i, ep in enumerate(tqdm(episodes, desc="Extracting")):
        try:
            # Build full text using Mistral format
            system_prompt = ep.get('system_prompt', '')
            user_turns = ep.get('user_turns', [])
            reply = ep.get('reply', '')

            # Format as Mistral conversation
            full_text = f"<s>[INST] {system_prompt}\n\n{user_turns[0]} [/INST]"
            for turn in user_turns[1:]:
                full_text += f"</s>[INST] {turn} [/INST]"
            full_text += reply

            # Tokenize
            token_ids = tokenizer.encode(full_text, add_special_tokens=False)

            # Find positions
            positions = find_token_positions(token_ids, tokenizer)
            probe_idx = get_safe_probe_index(positions, preference="before_tool")

            if probe_idx is None:
                skipped += 1
                continue

            # Now run forward pass
            inputs = tokenizer(full_text, return_tensors="pt", truncation=True, max_length=4096)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Extract activations
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            # Get activation from target layer at probe position
            # hidden_states[0] = embeddings, hidden_states[1] = layer 0, ..., hidden_states[17] = layer 16
            hidden_state = outputs.hidden_states[TARGET_LAYER + 1]  # +1 because index 0 is embeddings

            # Use probe position (before tool call)
            activation = hidden_state[0, probe_idx, :].cpu().numpy()

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
    print(f"  Skipped: {skipped} episodes (no valid probe position)")
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
