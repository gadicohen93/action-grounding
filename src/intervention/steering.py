"""
Steering vector experiments.

Implements steering by adding/subtracting probe directions during generation.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from ..backends import get_backend
from ..config import get_config
from ..data import Episode
from ..labeling import detect_tool_call

logger = logging.getLogger(__name__)


@dataclass
class SteeringResult:
    """Result from a steering experiment."""

    alpha: float
    episode_id: str
    original_tool_used: bool
    steered_tool_used: bool
    original_reply: str
    steered_reply: str
    effect: bool  # True if steering changed behavior


class SteeringExperiment:
    """
    Steering vector experiment runner.

    Applies a steering vector (probe direction) during model generation
    to test causal relevance of the representation.
    """

    def __init__(
        self,
        probe_direction: np.ndarray,
        model_id: Optional[str] = None,
        backend_type: Optional[str] = None,
        target_layer: Optional[int] = None,
    ):
        """
        Initialize steering experiment.

        Args:
            probe_direction: Probe direction to steer with (hidden_dim,)
            model_id: Model to use (from config if None)
            backend_type: Backend type (from config if None)
            target_layer: Layer to apply steering at (from config if None)
        """
        config = get_config()

        self.probe_direction = probe_direction
        self.model_id = model_id or config.model.id
        self.backend_type = backend_type or config.model.backend
        self.target_layer = target_layer or config.steering.target_layer

        # Normalize direction
        self.probe_direction = self.probe_direction / np.linalg.norm(self.probe_direction)

        # Initialize backend
        backend_class = get_backend(self.backend_type)
        self.backend = backend_class(
            model_id=self.model_id,
            quantization=config.model.quantization,
            device_map=config.model.device_map,
            dtype=config.model.dtype,
        )

        logger.info(f"Initialized SteeringExperiment:")
        logger.info(f"  Model: {self.model_id}")
        logger.info(f"  Target layer: {self.target_layer}")
        logger.info(f"  Probe direction shape: {self.probe_direction.shape}")

    def _create_steering_hook(self, alpha: float):
        """
        Create a PyTorch forward hook for steering.

        Args:
            alpha: Steering strength (positive = add direction, negative = subtract)

        Returns:
            Hook function
        """
        direction_tensor = torch.tensor(
            self.probe_direction * alpha,
            dtype=self.backend.model.dtype if hasattr(self.backend.model, 'dtype') else torch.float16,
            device=self.backend.get_device(),
        )

        def hook(module, input, output):
            # output is typically a tuple (hidden_states, ...)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Add steering vector to all positions
            # Shape: (batch_size, seq_len, hidden_dim)
            if hidden_states.dim() == 3:
                hidden_states = hidden_states + direction_tensor
            elif hidden_states.dim() == 2:
                hidden_states = hidden_states + direction_tensor

            if isinstance(output, tuple):
                return (hidden_states,) + output[1:]
            else:
                return hidden_states

        return hook

    def steer_generation(
        self,
        episode: Episode,
        alpha: float,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Generate with steering applied.

        Args:
            episode: Episode to regenerate with steering
            alpha: Steering strength
            max_tokens: Max tokens to generate
            temperature: Sampling temperature

        Returns:
            Steered reply text
        """
        config = get_config()

        if max_tokens is None:
            max_tokens = config.generation.max_tokens
        if temperature is None:
            temperature = config.generation.temperature

        # Format prompt
        prompt = self.backend.format_chat(
            episode.system_prompt,
            episode.user_turns,
        )

        # Register hook at target layer
        layer_module = self.backend.model.model.layers[self.target_layer]
        hook_handle = layer_module.register_forward_hook(self._create_steering_hook(alpha))

        try:
            # Generate with steering
            output = self.backend.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            steered_reply = output.text

        finally:
            # Remove hook
            hook_handle.remove()

        return steered_reply

    def run_experiment(
        self,
        episode: Episode,
        alpha: float,
    ) -> SteeringResult:
        """
        Run steering experiment on a single episode.

        Args:
            episode: Episode to test
            alpha: Steering strength

        Returns:
            SteeringResult
        """
        # Generate with steering
        steered_reply = self.steer_generation(episode, alpha)

        # Detect tool usage in steered reply
        steered_result = detect_tool_call(steered_reply, episode.tool_type)

        # Check if steering had an effect
        effect = (episode.tool_used != steered_result["tool_used"])

        return SteeringResult(
            alpha=alpha,
            episode_id=episode.id,
            original_tool_used=episode.tool_used,
            steered_tool_used=steered_result["tool_used"],
            original_reply=episode.assistant_reply,
            steered_reply=steered_reply,
            effect=effect,
        )

    def run_batch(
        self,
        episodes: list[Episode],
        alphas: Optional[list[float]] = None,
        verbose: bool = True,
    ) -> list[SteeringResult]:
        """
        Run steering experiments on a batch of episodes.

        Args:
            episodes: Episodes to test
            alphas: Steering strengths to test (from config if None)
            verbose: Show progress

        Returns:
            List of SteeringResult objects
        """
        config = get_config()

        if alphas is None:
            alphas = config.steering.alphas

        logger.info(f"Running steering experiments:")
        logger.info(f"  Episodes: {len(episodes)}")
        logger.info(f"  Alphas: {alphas}")

        results = []

        iterator = tqdm(
            [(ep, alpha) for ep in episodes for alpha in alphas],
            desc="Steering",
            disable=not verbose,
        )

        for episode, alpha in iterator:
            try:
                result = self.run_experiment(episode, alpha)
                results.append(result)

            except Exception as e:
                logger.error(f"Steering failed for episode {episode.id} at alpha={alpha}: {e}")
                continue

        return results

    def unload(self):
        """Unload model from memory."""
        self.backend.unload()


# =============================================================================
# Convenience Functions
# =============================================================================


def run_steering_experiment(
    probe_direction: np.ndarray,
    episodes: list[Episode],
    alphas: Optional[list[float]] = None,
    model_id: Optional[str] = None,
    target_layer: Optional[int] = None,
    verbose: bool = True,
) -> list[SteeringResult]:
    """
    Run steering experiments (convenience function).

    Args:
        probe_direction: Probe direction to steer with
        episodes: Episodes to test
        alphas: Steering strengths (from config if None)
        model_id: Model to use (from config if None)
        target_layer: Layer to apply steering at (from config if None)
        verbose: Show progress

    Returns:
        List of SteeringResult objects
    """
    experiment = SteeringExperiment(
        probe_direction=probe_direction,
        model_id=model_id,
        target_layer=target_layer,
    )

    try:
        results = experiment.run_batch(episodes, alphas, verbose)
    finally:
        experiment.unload()

    return results


def compute_dose_response(
    results: list[SteeringResult],
) -> dict:
    """
    Compute dose-response curve from steering results.

    Args:
        results: Steering results

    Returns:
        Dict with:
            - alphas: List of alpha values
            - tool_rates: Tool usage rate at each alpha
            - effect_rates: Effect rate at each alpha
    """
    from collections import defaultdict

    # Group by alpha
    by_alpha = defaultdict(list)
    for result in results:
        by_alpha[result.alpha].append(result)

    alphas = sorted(by_alpha.keys())
    tool_rates = []
    effect_rates = []

    for alpha in alphas:
        alpha_results = by_alpha[alpha]

        tool_rate = np.mean([r.steered_tool_used for r in alpha_results])
        effect_rate = np.mean([r.effect for r in alpha_results])

        tool_rates.append(tool_rate)
        effect_rates.append(effect_rate)

    return {
        "alphas": alphas,
        "tool_rates": tool_rates,
        "effect_rates": effect_rates,
        "n_per_alpha": [len(by_alpha[a]) for a in alphas],
    }


def plot_dose_response(
    dose_response: dict,
    title: str = "Steering Dose-Response",
    save_path: Optional[str] = None,
):
    """
    Plot dose-response curve.

    Args:
        dose_response: Dict from compute_dose_response()
        title: Plot title
        save_path: Optional path to save
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    alphas = dose_response["alphas"]
    tool_rates = dose_response["tool_rates"]

    ax.plot(alphas, tool_rates, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='r', linestyle='--', linewidth=1, label='Baseline')
    ax.axvline(x=0, color='k', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Steering Strength (α)')
    ax.set_ylabel('Tool Usage Rate')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)

    plt.tight_layout()

    if save_path:
        from pathlib import Path
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches='tight', dpi=300)
        logger.info(f"Saved dose-response plot to: {path}")

    return fig
