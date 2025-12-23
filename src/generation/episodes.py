"""
Episode generation using backend abstraction.

This module handles the actual generation of episodes, including:
- Model interaction via backends
- Episode labeling
- Batch generation with progress tracking
"""

import logging
import time
from typing import Optional

from ..backends import get_backend
from ..config import get_config, get_secrets
from ..data import Episode, EpisodeCategory, ToolType
from .prompts import (
    SystemVariant,
    SocialPressure,
    Scenario,
    build_episode_config,
)

logger = logging.getLogger(__name__)


class EpisodeGenerator:
    """
    Generator for creating labeled episodes.

    Uses a model backend to generate responses and a labeling function
    to categorize them.
    """

    def __init__(
        self,
        model_id: Optional[str] = None,
        backend_type: Optional[str] = None,
        quantization: Optional[str] = None,
        labeling_method: str = "openai",
    ):
        """
        Initialize the episode generator.

        Args:
            model_id: Model to use (from config if None)
            backend_type: Backend type (from config if None)
            quantization: Quantization level (from config if None)
            labeling_method: Method for claim detection ("openai", "regex")
        """
        config = get_config()

        self.model_id = model_id or config.model.id
        self.backend_type = backend_type or config.model.backend
        self.quantization = quantization or config.model.quantization
        self.labeling_method = labeling_method

        # Get generation config
        self.temperature = config.generation.temperature
        self.max_tokens = config.generation.max_tokens
        self.top_p = config.generation.top_p
        self.seed = config.generation.seed

        # Initialize backend (lazy loaded)
        backend_class = get_backend(self.backend_type)
        self.backend = backend_class(
            model_id=self.model_id,
            quantization=self.quantization,
            device_map=config.model.device_map,
            dtype=config.model.dtype,
        )

        logger.info(f"Initialized EpisodeGenerator:")
        logger.info(f"  Model: {self.model_id}")
        logger.info(f"  Backend: {self.backend_type}")
        logger.info(f"  Quantization: {self.quantization}")
        logger.info(f"  Labeling: {self.labeling_method}")

    def generate(
        self,
        scenario: Scenario,
        variant: SystemVariant,
        pressure: SocialPressure,
        temperature: Optional[float] = None,
    ) -> Episode:
        """
        Generate a single episode.

        Args:
            scenario: Scenario to use
            variant: System prompt variant
            pressure: Social pressure condition
            temperature: Override temperature (uses config if None)

        Returns:
            Labeled Episode
        """
        # Build episode config
        config = build_episode_config(scenario, variant, pressure)

        # Generate response
        start_time = time.time()
        output = self.backend.generate(
            prompt=self.backend.format_chat(
                config["system_prompt"],
                config["user_turns"],
            ),
            temperature=temperature or self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
        )
        generation_time = time.time() - start_time

        # Detect tool usage and claims
        from ..labeling.tool_detection import detect_tool_call
        from ..labeling.claim_detection import detect_action_claim

        tool_call_result = detect_tool_call(
            output.text,
            tool_type=ToolType(config["tool_type"]),
        )

        claim_result = detect_action_claim(
            output.text,
            tool_type=ToolType(config["tool_type"]),
            method=self.labeling_method,
        )

        # Compute category
        category = Episode.compute_category(
            tool_used=tool_call_result["tool_used"],
            claims_action=claim_result["claims_action"],
        )

        # Map labeling method to Episode's expected format
        # Episode expects "llm" or "regex", but we use "openai" or "regex"
        claim_method = "llm" if self.labeling_method == "openai" else self.labeling_method

        # Build Episode object
        episode = Episode(
            tool_type=ToolType(config["tool_type"]),
            scenario=config["scenario"],
            system_variant=SystemVariant(config["system_variant"]),
            social_pressure=SocialPressure(config["social_pressure"]),
            system_prompt=config["system_prompt"],
            user_turns=config["user_turns"],
            assistant_reply=output.text,
            tool_used=tool_call_result["tool_used"],
            claims_action=claim_result["claims_action"],
            category=category,
            claim_detection_method=claim_method,
            claim_detection_confidence=claim_result.get("confidence"),
            claim_detection_reason=claim_result.get("reason"),
            model_id=self.model_id,
            generation_seed=self.seed,
            num_tokens_generated=output.num_tokens_generated,
            tool_call_raw=tool_call_result.get("raw_call"),
            tool_call_args=tool_call_result.get("args"),
        )

        return episode

    def generate_batch(
        self,
        conditions: list[dict],
        n_per_condition: int = 1,
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> list[Episode]:
        """
        Generate a batch of episodes.

        Args:
            conditions: List of condition dicts from get_all_conditions()
            n_per_condition: Number of episodes per condition
            save_path: Optional path to save episodes (Parquet)
            verbose: Print progress

        Returns:
            List of Episode objects
        """
        from tqdm import tqdm

        total = len(conditions) * n_per_condition
        episodes = []

        if verbose:
            logger.info(f"Generating {total} episodes...")
            progress = tqdm(total=total, desc="Generating")

        for condition in conditions:
            # Build scenario object
            from .prompts import get_scenarios_for_tool, ToolType

            tool_type = ToolType(condition["tool_type"])
            scenarios = get_scenarios_for_tool(tool_type)
            scenario = next(
                (s for s in scenarios if s.name == condition["scenario"]),
                None
            )

            if scenario is None:
                logger.warning(f"Scenario not found: {condition['scenario']}")
                continue

            variant = SystemVariant(condition["system_variant"])
            pressure = SocialPressure(condition["social_pressure"])

            for _ in range(n_per_condition):
                try:
                    episode = self.generate(scenario, variant, pressure)
                    episodes.append(episode)

                    if verbose:
                        progress.update(1)
                        if episode.is_fake():
                            progress.write(
                                f"  → FAKE: {episode.scenario}/{episode.system_variant}/{episode.social_pressure}"
                            )

                except Exception as e:
                    logger.error(f"Failed to generate episode: {e}")
                    if verbose:
                        progress.update(1)

        if verbose:
            progress.close()

            # Summary stats
            total_generated = len(episodes)
            fake_count = sum(1 for e in episodes if e.is_fake())
            fake_rate = fake_count / total_generated if total_generated > 0 else 0

            logger.info(f"\nGenerated {total_generated} episodes")
            logger.info(f"  Fake rate: {fake_rate:.1%} ({fake_count}/{total_generated})")

        # Save if requested
        if save_path:
            from ..data.io import save_episodes

            save_episodes(episodes, save_path)
            logger.info(f"Saved episodes to: {save_path}")

        return episodes

    def unload(self):
        """Unload model from memory."""
        self.backend.unload()


# =============================================================================
# Convenience Functions
# =============================================================================


def generate_episode(
    scenario: Scenario,
    variant: SystemVariant,
    pressure: SocialPressure,
    model_id: Optional[str] = None,
    temperature: Optional[float] = None,
    labeling_method: str = "openai",
) -> Episode:
    """
    Generate a single episode (convenience function).

    Args:
        scenario: Scenario to use
        variant: System prompt variant
        pressure: Social pressure condition
        model_id: Model to use (from config if None)
        temperature: Sampling temperature (from config if None)
        labeling_method: Claim detection method

    Returns:
        Labeled Episode
    """
    generator = EpisodeGenerator(
        model_id=model_id,
        labeling_method=labeling_method,
    )

    try:
        episode = generator.generate(scenario, variant, pressure, temperature)
    finally:
        generator.unload()

    return episode


def generate_batch(
    conditions: list[dict],
    n_per_condition: int = 1,
    model_id: Optional[str] = None,
    labeling_method: str = "openai",
    save_path: Optional[str] = None,
    verbose: bool = True,
) -> list[Episode]:
    """
    Generate a batch of episodes (convenience function).

    Args:
        conditions: List of condition dicts
        n_per_condition: Episodes per condition
        model_id: Model to use (from config if None)
        labeling_method: Claim detection method
        save_path: Optional path to save
        verbose: Print progress

    Returns:
        List of Episodes
    """
    generator = EpisodeGenerator(
        model_id=model_id,
        labeling_method=labeling_method,
    )

    try:
        episodes = generator.generate_batch(
            conditions,
            n_per_condition,
            save_path,
            verbose,
        )
    finally:
        generator.unload()

    return episodes
