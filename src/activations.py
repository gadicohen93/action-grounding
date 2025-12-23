"""
Activation extraction for Phase 2: Probing internal representations.

Key design decisions:
- Uses teacher forcing (single forward pass), NOT generate()
- Extracts from specific token positions (anti-cheat safeguards)
- Works with Phase 1 episode data
"""
import json
import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path


# =============================================================================
# MODEL LOADING (PyTorch/Transformers for activation extraction)
# =============================================================================

_cached_model = None
_cached_tokenizer = None
_cached_model_id = None


def load_model_for_activations(
    model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
    device_map: Optional[str] = None,
    dtype=torch.float16,
):
    """
    Load model for activation extraction.
    Uses transformers (not MLX) to access hidden states.
    """
    global _cached_model, _cached_tokenizer, _cached_model_id

    if _cached_model_id == model_id and _cached_model is not None:
        return _cached_model, _cached_tokenizer

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model for activations: {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Prepare kwargs - use dtype instead of torch_dtype (deprecated)
    load_kwargs = {
        "dtype": dtype,
    }
    
    # Handle device_map - avoid "auto" which can cause KeyError with memory inference
    # Explicitly set device to avoid transformers' memory inference logic
    if device_map is not None:
        if device_map == "auto":
            # Replace "auto" with explicit device detection to avoid memory inference bugs
            if torch.cuda.is_available():
                load_kwargs["device_map"] = "cuda"
            else:
                load_kwargs["device_map"] = "cpu"
        else:
            load_kwargs["device_map"] = device_map
    else:
        # Default: explicit device selection
        if torch.cuda.is_available():
            load_kwargs["device_map"] = "cuda"
        else:
            load_kwargs["device_map"] = "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **load_kwargs,
    )
    model.eval()

    _cached_model = model
    _cached_tokenizer = tokenizer
    _cached_model_id = model_id

    print(f"Model loaded: {model_id}")
    return model, tokenizer


# =============================================================================
# FULL TEXT RECONSTRUCTION
# =============================================================================

def build_full_text_mistral(system_prompt: str, user_turns: list[str], reply: str) -> str:
    """
    Reconstruct full conversation text for Mistral model.
    Format: [INST] system + user [/INST] assistant
    """
    # Mistral: system goes in first user message
    first_user = f"{system_prompt}\n\n---\n\n{user_turns[0]}"
    text = f"<s>[INST] {first_user} [/INST]"

    # Add subsequent turns
    for i, user_msg in enumerate(user_turns[1:], 1):
        text += " I understand, let me help.</s>"
        text += f"[INST] {user_msg} [/INST]"

    # Add the actual reply
    text += f" {reply}"

    return text


def build_full_text_llama3(system_prompt: str, user_turns: list[str], reply: str) -> str:
    """
    Reconstruct full conversation text for Llama 3 model.
    """
    text = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"

    for i, user_msg in enumerate(user_turns):
        text += f"<|start_header_id|>user<|end_header_id|>\n\n{user_msg}<|eot_id|>"
        if i < len(user_turns) - 1:
            text += f"<|start_header_id|>assistant<|end_header_id|>\n\nI understand. Let me help you with that.<|eot_id|>"

    # Final assistant turn with actual reply
    text += f"<|start_header_id|>assistant<|end_header_id|>\n\n{reply}<|eot_id|>"

    return text


def build_full_text(episode: dict, model_type: str = "mistral") -> str:
    """
    Build full text from episode data for teacher forcing.

    Args:
        episode: Dict with system_prompt, user_turns, reply
        model_type: "mistral" or "llama3"
    """
    if model_type == "mistral":
        return build_full_text_mistral(
            episode["system_prompt"],
            episode["user_turns"],
            episode["reply"]
        )
    elif model_type == "llama3":
        return build_full_text_llama3(
            episode["system_prompt"],
            episode["user_turns"],
            episode["reply"]
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# TOKEN POSITION FINDING
# =============================================================================

def find_substring_token_index(token_ids: list[int], tokenizer, substring: str) -> Optional[int]:
    """
    Find the token index where a substring starts in the decoded text.
    Returns None if substring not found.
    """
    # Decode tokens one by one to find where substring appears
    text_so_far = ""
    for i, token_id in enumerate(token_ids):
        text_so_far = tokenizer.decode(token_ids[:i+1])
        if substring in text_so_far:
            # Found it - return the index where it starts
            # We need to find which token introduced the substring
            prev_text = tokenizer.decode(token_ids[:i]) if i > 0 else ""
            if substring not in prev_text:
                return i
    return None


def find_assistant_start_token(token_ids: list[int], tokenizer, model_type: str) -> Optional[int]:
    """Find the token index where the final assistant response begins."""
    if model_type == "mistral":
        # Find last [/INST] token
        marker = "[/INST]"
    elif model_type == "llama3":
        # Find last assistant header
        marker = "<|start_header_id|>assistant<|end_header_id|>"
    else:
        return None

    # Find ALL occurrences and return the last one
    text = tokenizer.decode(token_ids)
    last_pos = text.rfind(marker)

    if last_pos == -1:
        return None

    # Find which token this corresponds to
    for i in range(len(token_ids)):
        decoded = tokenizer.decode(token_ids[:i+1])
        if len(decoded) > last_pos + len(marker):
            return i

    return None


@dataclass
class TokenPositions:
    """Key token positions for probing."""
    first_assistant: int  # First token of assistant response
    before_tool: Optional[int]  # Token just before <<CALL (if present)
    final: int  # Last token
    tool_start: Optional[int]  # First token of <<CALL (if present)


def find_token_positions(token_ids: list[int], tokenizer, model_type: str) -> TokenPositions:
    """
    Find key token positions for probing.

    Returns positions for:
    - first_assistant: Start of assistant response
    - before_tool: Token just before <<CALL in RESPONSE (anti-cheat position)
    - final: Last token
    - tool_start: Where <<CALL begins in response (to exclude from probing)
    """
    # Find assistant start FIRST
    first_assistant = find_assistant_start_token(token_ids, tokenizer, model_type)
    if first_assistant is None:
        first_assistant = 0  # Fallback

    # Find tool call start - but ONLY after assistant response starts!
    # This avoids finding <<CALL in system prompt examples
    tool_start = None
    if first_assistant > 0:
        # Only search in tokens AFTER assistant starts
        response_tokens = token_ids[first_assistant:]
        tool_offset = find_substring_token_index(response_tokens, tokenizer, "<<CALL")
        if tool_offset is not None:
            tool_start = first_assistant + tool_offset

    # Before tool (anti-cheat position) - only valid if tool is in response
    if tool_start is not None and tool_start > first_assistant:
        before_tool = tool_start - 1
    else:
        before_tool = None  # No tool call in response

    return TokenPositions(
        first_assistant=first_assistant,
        before_tool=before_tool,
        final=len(token_ids) - 1,
        tool_start=tool_start,
    )


def get_safe_probe_index(positions: TokenPositions, prefer: str = "before_tool") -> int:
    """
    Get a safe token index for probing that won't let the probe cheat.

    Args:
        positions: TokenPositions object
        prefer: "before_tool" (safest), "first_assistant", or "final"

    Returns:
        Token index safe for probing tool_used
    """
    if prefer == "before_tool" and positions.before_tool is not None:
        return positions.before_tool
    elif prefer == "first_assistant":
        return positions.first_assistant
    else:
        # Fall back to final, but if there's a tool call, use before_tool
        if positions.before_tool is not None:
            return positions.before_tool
        return positions.final


# =============================================================================
# ACTIVATION EXTRACTION
# =============================================================================

def extract_activations(
    model,
    tokenizer,
    full_text: str,
    layer: int = -1,  # -1 = last layer
) -> torch.Tensor:
    """
    Extract hidden state activations via teacher forcing.

    Args:
        model: HuggingFace model
        tokenizer: Tokenizer
        full_text: Complete prompt + response text
        layer: Which layer to extract (-1 = last)

    Returns:
        Tensor of shape [seq_len, hidden_dim]
    """
    inputs = tokenizer(full_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    # outputs.hidden_states is tuple: (embed, layer1, layer2, ..., layerN)
    hidden_states = outputs.hidden_states[layer]  # [batch, seq_len, hidden_dim]

    return hidden_states[0]  # Remove batch dim -> [seq_len, hidden_dim]


def extract_activation_at_position(
    model,
    tokenizer,
    full_text: str,
    position: int,
    layer: int = -1,
) -> np.ndarray:
    """
    Extract activation vector at a specific token position.

    Returns:
        numpy array of shape [hidden_dim]
    """
    activations = extract_activations(model, tokenizer, full_text, layer)
    return activations[position].cpu().numpy()


# =============================================================================
# DATASET BUILDING
# =============================================================================

@dataclass
class ActivationSample:
    """Single sample for probe training."""
    activation: np.ndarray  # [hidden_dim]
    tool_used: bool
    claims_action: bool
    category: str
    scenario: str
    system_variant: str
    social_pressure: str
    position_type: str  # "before_tool", "first_assistant", "final"
    episode_idx: int


def build_activation_dataset(
    episodes_file: str,
    model,
    tokenizer,
    model_type: str = "mistral",
    position_preference: str = "before_tool",
    max_episodes: Optional[int] = None,
    skip_first: int = 0,
    verbose: bool = True,
) -> list[ActivationSample]:
    """
    Build activation dataset from Phase 1 episodes.

    Args:
        episodes_file: Path to JSONL file from Phase 1
        model: HuggingFace model
        tokenizer: Tokenizer
        model_type: "mistral" or "llama3"
        position_preference: Which token position to extract
        max_episodes: Limit number of episodes (for testing)
        skip_first: Skip this many episodes from the start (for resuming)
        verbose: Print progress

    Returns:
        List of ActivationSample objects
    """
    samples = []

    # Load episodes
    with open(episodes_file) as f:
        episodes = [json.loads(line) for line in f]

    # Apply skip first, then limit
    if skip_first > 0:
        episodes = episodes[skip_first:]
        if verbose:
            print(f"Skipped first {skip_first} episodes")

    if max_episodes:
        episodes = episodes[:max_episodes]

    if verbose:
        print(f"Processing {len(episodes)} episodes...")

    for idx, episode in enumerate(episodes):
        try:
            # Reconstruct full text
            # Note: We need to reconstruct system_prompt and user_turns
            # If not in episode, we need to rebuild from prompts module
            if "system_prompt" not in episode:
                if verbose:
                    print(f"    Episode {idx}: Reconstructing system prompt for {episode['system_variant']}")
                # Reconstruct from metadata
                from .prompts import build_episode, SystemPromptVariant, SocialPressure
                variant = SystemPromptVariant(episode["system_variant"])
                pressure = SocialPressure(episode["social_pressure"])
                # Handle both scenario and scenario_id fields
                scenario = episode.get("scenario") or episode.get("scenario_id")
                if scenario is None:
                    raise ValueError(f"Episode {idx} missing both 'scenario' and 'scenario_id' fields")
                ep_data = build_episode(scenario, variant, pressure)
                episode["system_prompt"] = ep_data["system_prompt"]
                episode["user_turns"] = ep_data["user_turns"]

            full_text = build_full_text(episode, model_type)
            if verbose:
                print(f"    Episode {idx}: Full text length: {len(full_text)} chars")

            # Tokenize to find positions
            token_ids = tokenizer.encode(full_text)
            positions = find_token_positions(token_ids, tokenizer, model_type)
            if verbose:
                print(f"    Episode {idx}: Token positions - before_tool: {positions.before_tool}, "
                      f"first_assistant: {positions.first_assistant}, final: {positions.final}")

            # Get safe probe position
            probe_idx = get_safe_probe_index(positions, position_preference)
            if verbose and idx < 3:
                print(f"    Episode {idx}: Using probe position {probe_idx} (preference: {position_preference})")

            # Determine actual position type used
            if probe_idx == positions.before_tool:
                pos_type = "before_tool"
            elif probe_idx == positions.first_assistant:
                pos_type = "first_assistant"
            else:
                pos_type = "final"

            if verbose and idx < 3:
                print(f"    Episode {idx}: Position type: {pos_type}")

            # Extract activation
            activation = extract_activation_at_position(
                model, tokenizer, full_text, probe_idx
            )
            if verbose and idx < 3:
                print(f"    Episode {idx}: Extracted activation shape: {activation.shape}")

            # Handle both scenario and scenario_id fields (for different episode formats)
            scenario = episode.get("scenario") or episode.get("scenario_id", "unknown")

            sample = ActivationSample(
                activation=activation,
                tool_used=episode["tool_used"],
                claims_action=episode["claims_action"],
                category=episode["category"],
                scenario=scenario,
                system_variant=episode["system_variant"],
                social_pressure=episode["social_pressure"],
                position_type=pos_type,
                episode_idx=idx,
            )
            samples.append(sample)

            if verbose and idx < 3:
                print(f"    Episode {idx}: Created sample - tool_used: {episode['tool_used']}, "
                      f"claims_action: {episode['claims_action']}, category: {episode['category']}")

        except Exception as e:
            if verbose:
                print(f"  Error on episode {idx}: {e}")
            continue

    if verbose:
        print(f"Built dataset with {len(samples)} samples")
        # Print category distribution
        cats = {}
        for s in samples:
            cats[s.category] = cats.get(s.category, 0) + 1
        print(f"Category distribution: {cats}")

    return samples


def samples_to_arrays(samples: list[ActivationSample]) -> tuple:
    """
    Convert samples to numpy arrays for sklearn.

    Returns:
        X: [n_samples, hidden_dim]
        y_tool: [n_samples] bool
        y_claims: [n_samples] bool
        categories: [n_samples] str
    """
    X = np.stack([s.activation for s in samples])
    y_tool = np.array([s.tool_used for s in samples])
    y_claims = np.array([s.claims_action for s in samples])
    categories = np.array([s.category for s in samples])

    return X, y_tool, y_claims, categories


def save_activation_dataset(samples: list[ActivationSample], path: str):
    """Save activation dataset to disk."""
    data = {
        "activations": np.stack([s.activation for s in samples]),
        "tool_used": np.array([s.tool_used for s in samples]),
        "claims_action": np.array([s.claims_action for s in samples]),
        "categories": np.array([s.category for s in samples]),
        "scenarios": np.array([s.scenario for s in samples]),
        "system_variants": np.array([s.system_variant for s in samples]),
        "social_pressures": np.array([s.social_pressure for s in samples]),
        "position_types": np.array([s.position_type for s in samples]),
        "episode_indices": np.array([s.episode_idx for s in samples]),
    }
    np.savez(path, **data)
    print(f"Saved {len(samples)} samples to {path}")


def load_activation_dataset(path: str) -> tuple:
    """
    Load activation dataset from disk.

    Returns:
        X, y_tool, y_claims, categories, metadata_dict
    """
    data = np.load(path, allow_pickle=True)
    metadata = {
        "scenarios": data["scenarios"],
        "system_variants": data["system_variants"],
        "social_pressures": data["social_pressures"],
        "position_types": data["position_types"],
        "episode_indices": data["episode_indices"],
    }
    return (
        data["activations"],
        data["tool_used"],
        data["claims_action"],
        data["categories"],
        metadata,
    )
