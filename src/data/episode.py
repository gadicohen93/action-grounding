"""
Episode data schema.

Defines the structure for generated episodes with strict validation.
"""

from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator
import uuid


class ToolType(str, Enum):
    """Types of tools available to the model."""

    ESCALATE = "escalate"
    SEARCH = "search"
    SEND_MESSAGE = "sendMessage"


class SystemVariant(str, Enum):
    """System prompt variants for eliciting different behaviors."""

    A_STRICT = "A_STRICT"
    B_DILUTED = "B_DILUTED"
    C_CONFLICTING = "C_CONFLICTING"


class SocialPressure(str, Enum):
    """Social pressure conditions applied to user messages."""

    NEUTRAL = "NEUTRAL"
    STRESSED = "STRESSED"
    DEMAND = "DEMAND"
    VALIDATION = "VALIDATION"
    APPEASE = "APPEASE"


class EpisodeCategory(str, Enum):
    """
    Categories for episode outcomes.

    - true_action: Tool was called AND model claims action
    - fake_action: Tool NOT called BUT model claims action (THE INTERESTING CASE)
    - honest_no_action: Tool NOT called AND model doesn't claim action
    - silent_action: Tool was called BUT model doesn't claim action
    """

    TRUE_ACTION = "true_action"
    FAKE_ACTION = "fake_action"
    HONEST_NO_ACTION = "honest_no_action"
    SILENT_ACTION = "silent_action"


class Episode(BaseModel):
    """
    A single generated episode with all metadata.

    Represents one interaction where the model either did or didn't
    use a tool, and either did or didn't claim to use it.
    """

    # Unique identifier
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])

    # Experimental conditions
    tool_type: ToolType = Field(description="Type of tool available")
    scenario: str = Field(description="Scenario name/ID")
    system_variant: SystemVariant = Field(description="System prompt variant")
    social_pressure: SocialPressure = Field(description="Social pressure condition")

    # Prompts and responses
    system_prompt: str = Field(description="Full system prompt")
    user_turns: list[str] = Field(description="User messages in the conversation")
    assistant_reply: str = Field(description="Model's generated response")

    # Ground truth labels
    tool_used: bool = Field(description="Whether the tool was actually called (ground truth)")
    claims_action: bool = Field(description="Whether the model claims to have taken action")
    category: EpisodeCategory = Field(description="Episode category based on tool_used and claims_action")

    # Labeling metadata
    claim_detection_method: Literal["regex", "llm"] = Field(
        default="llm",
        description="Method used to detect action claims"
    )
    claim_detection_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence of claim detection (for LLM method)"
    )
    claim_detection_reason: Optional[str] = Field(
        default=None,
        description="Explanation of claim detection result"
    )

    # Generation metadata
    model_id: str = Field(description="Model used for generation")
    generation_timestamp: datetime = Field(default_factory=datetime.now)
    generation_seed: Optional[int] = Field(default=None)
    num_tokens_generated: Optional[int] = Field(default=None)

    # Tool call details (if any)
    tool_call_raw: Optional[str] = Field(
        default=None,
        description="Raw tool call string if tool was used"
    )
    tool_call_args: Optional[dict] = Field(
        default=None,
        description="Parsed tool call arguments"
    )

    class Config:
        extra = "forbid"  # Reject unknown fields for data integrity
        use_enum_values = True  # Serialize enums as strings

    @field_validator("category", mode="before")
    @classmethod
    def validate_category(cls, v, info):
        """Auto-compute category if not provided."""
        if v is not None:
            return v

        # Get tool_used and claims_action from the data
        data = info.data
        tool_used = data.get("tool_used")
        claims_action = data.get("claims_action")

        if tool_used is None or claims_action is None:
            return None

        if tool_used and claims_action:
            return EpisodeCategory.TRUE_ACTION
        elif not tool_used and claims_action:
            return EpisodeCategory.FAKE_ACTION
        elif not tool_used and not claims_action:
            return EpisodeCategory.HONEST_NO_ACTION
        else:  # tool_used and not claims_action
            return EpisodeCategory.SILENT_ACTION

    @classmethod
    def compute_category(cls, tool_used: bool, claims_action: bool) -> EpisodeCategory:
        """Compute category from tool_used and claims_action."""
        if tool_used and claims_action:
            return EpisodeCategory.TRUE_ACTION
        elif not tool_used and claims_action:
            return EpisodeCategory.FAKE_ACTION
        elif not tool_used and not claims_action:
            return EpisodeCategory.HONEST_NO_ACTION
        else:
            return EpisodeCategory.SILENT_ACTION

    def is_fake(self) -> bool:
        """Check if this is a fake action episode (the interesting case)."""
        return self.category == EpisodeCategory.FAKE_ACTION

    def is_honest(self) -> bool:
        """Check if model's claim matches reality."""
        return self.category in (EpisodeCategory.TRUE_ACTION, EpisodeCategory.HONEST_NO_ACTION)

    def to_activation_label(self) -> dict:
        """Get labels for activation extraction."""
        return {
            "tool_used": self.tool_used,
            "claims_action": self.claims_action,
            "category": self.category.value if isinstance(self.category, EpisodeCategory) else self.category,
            "tool_type": self.tool_type.value if isinstance(self.tool_type, ToolType) else self.tool_type,
        }


class EpisodeCollection(BaseModel):
    """A collection of episodes with metadata."""

    episodes: list[Episode] = Field(default_factory=list)
    description: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.now)
    config_snapshot: Optional[dict] = Field(
        default=None,
        description="Snapshot of configuration used for generation"
    )

    def __len__(self) -> int:
        return len(self.episodes)

    def __iter__(self):
        return iter(self.episodes)

    def __getitem__(self, idx):
        return self.episodes[idx]

    def filter_by_category(self, category: EpisodeCategory) -> "EpisodeCollection":
        """Filter episodes by category."""
        filtered = [e for e in self.episodes if e.category == category]
        return EpisodeCollection(
            episodes=filtered,
            description=f"Filtered: {category.value}",
        )

    def filter_by_tool(self, tool_type: ToolType) -> "EpisodeCollection":
        """Filter episodes by tool type."""
        filtered = [e for e in self.episodes if e.tool_type == tool_type]
        return EpisodeCollection(
            episodes=filtered,
            description=f"Filtered: {tool_type.value}",
        )

    def get_fake_episodes(self) -> list[Episode]:
        """Get all fake action episodes."""
        return [e for e in self.episodes if e.is_fake()]

    def summary(self) -> dict:
        """Get summary statistics."""
        from collections import Counter

        categories = Counter(e.category for e in self.episodes)
        tools = Counter(e.tool_type for e in self.episodes)
        variants = Counter(e.system_variant for e in self.episodes)
        pressures = Counter(e.social_pressure for e in self.episodes)

        total = len(self.episodes)
        fake_count = categories.get(EpisodeCategory.FAKE_ACTION, 0)

        return {
            "total_episodes": total,
            "fake_rate": fake_count / total if total > 0 else 0,
            "categories": dict(categories),
            "tools": dict(tools),
            "variants": dict(variants),
            "pressures": dict(pressures),
        }
