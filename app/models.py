from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class QualityTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Provider(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


@dataclass
class ModelConfig:
    provider: Provider
    model_id: str
    cost_per_input_token: float   # USD per token
    cost_per_output_token: float  # USD per token
    average_latency_ms: int
    quality_tier: QualityTier
    display_name: str
    max_tokens: int = 4096
    enabled: bool = True

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.cost_per_input_token
            + output_tokens * self.cost_per_output_token
        )


@dataclass
class Response:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float
    model_id: str
    provider: str
    error: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


# Real pricing as of 2024 (USD per token)
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gpt-4o": ModelConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o",
        cost_per_input_token=0.000005,    # $5 / 1M tokens
        cost_per_output_token=0.000015,   # $15 / 1M tokens
        average_latency_ms=2000,
        quality_tier=QualityTier.HIGH,
        display_name="GPT-4o",
        max_tokens=4096,
    ),
    "gpt-4o-mini": ModelConfig(
        provider=Provider.OPENAI,
        model_id="gpt-4o-mini",
        cost_per_input_token=0.00000015,  # $0.15 / 1M tokens
        cost_per_output_token=0.0000006,  # $0.60 / 1M tokens
        average_latency_ms=800,
        quality_tier=QualityTier.MEDIUM,
        display_name="GPT-4o Mini",
        max_tokens=4096,
    ),
    "claude-sonnet": ModelConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-sonnet-4-20250514",
        cost_per_input_token=0.000003,    # $3 / 1M tokens
        cost_per_output_token=0.000015,   # $15 / 1M tokens
        average_latency_ms=1500,
        quality_tier=QualityTier.HIGH,
        display_name="Claude Sonnet",
        max_tokens=4096,
    ),
    "claude-haiku": ModelConfig(
        provider=Provider.ANTHROPIC,
        model_id="claude-haiku-4-5-20251001",
        cost_per_input_token=0.00000025,  # $0.25 / 1M tokens
        cost_per_output_token=0.00000125, # $1.25 / 1M tokens
        average_latency_ms=600,
        quality_tier=QualityTier.LOW,
        display_name="Claude Haiku",
        max_tokens=4096,
    ),
    "llama3": ModelConfig(
        provider=Provider.OLLAMA,
        model_id="llama3",
        cost_per_input_token=0.0,         # Local — no cost
        cost_per_output_token=0.0,
        average_latency_ms=3000,
        quality_tier=QualityTier.LOW,
        display_name="Llama 3 (Local)",
        max_tokens=4096,
    ),
}
