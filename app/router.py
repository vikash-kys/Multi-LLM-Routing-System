import logging
from pathlib import Path
from typing import Optional

import yaml

from app.models import ModelConfig, MODEL_REGISTRY, QualityTier
from app.classifier import ComplexityTier, get_classifier

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("routing_config.yaml")

# Default routing: tier -> model registry key
DEFAULT_ROUTING = {
    ComplexityTier.SIMPLE: "claude-haiku",
    ComplexityTier.MODERATE: "gpt-4o-mini",
    ComplexityTier.COMPLEX: "gpt-4o",
}

# The highest-quality model used for verification
VERIFIER_MODEL_KEY = "gpt-4o"


def load_routing_config() -> dict[ComplexityTier, str]:
    """Load tier-to-model mapping from YAML. Falls back to defaults."""
    if not CONFIG_PATH.exists():
        return DEFAULT_ROUTING.copy()

    try:
        with open(CONFIG_PATH) as f:
            raw = yaml.safe_load(f)

        routing_raw = raw.get("routing", {})
        mapping = {}
        tier_map = {"tier_1": ComplexityTier.SIMPLE, "tier_2": ComplexityTier.MODERATE, "tier_3": ComplexityTier.COMPLEX}

        for yaml_key, tier in tier_map.items():
            if yaml_key in routing_raw:
                entry = routing_raw[yaml_key]
                # Find matching model in registry by provider + model_id
                model_key = _find_model_key(entry.get("provider"), entry.get("model_id"))
                if model_key:
                    mapping[tier] = model_key
                else:
                    mapping[tier] = DEFAULT_ROUTING[tier]
            else:
                mapping[tier] = DEFAULT_ROUTING[tier]

        return mapping

    except Exception as e:
        logger.warning(f"Could not load routing config: {e}. Using defaults.")
        return DEFAULT_ROUTING.copy()


def save_routing_config(mapping: dict) -> None:
    """Persist updated routing config to YAML."""
    # Read existing config to preserve quality settings
    existing = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            existing = yaml.safe_load(f) or {}

    tier_yaml_keys = {
        ComplexityTier.SIMPLE: "tier_1",
        ComplexityTier.MODERATE: "tier_2",
        ComplexityTier.COMPLEX: "tier_3",
    }

    routing_section = {}
    for tier, model_key in mapping.items():
        model = MODEL_REGISTRY.get(model_key)
        if model:
            routing_section[tier_yaml_keys[tier]] = {
                "provider": model.provider.value,
                "model_id": model.model_id,
            }

    existing["routing"] = routing_section
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(existing, f, default_flow_style=False)


def _find_model_key(provider: Optional[str], model_id: Optional[str]) -> Optional[str]:
    for key, config in MODEL_REGISTRY.items():
        if config.provider.value == provider and config.model_id == model_id:
            return key
        # Also allow matching by registry key directly
        if key == model_id:
            return key
    return None


def get_quality_config() -> dict:
    """Load quality thresholds from YAML."""
    if not CONFIG_PATH.exists():
        return {"min_judge_score": 4, "auto_escalate": True, "retraining_schedule": "weekly"}

    try:
        with open(CONFIG_PATH) as f:
            raw = yaml.safe_load(f)
        return raw.get("quality", {"min_judge_score": 4, "auto_escalate": True})
    except Exception:
        return {"min_judge_score": 4, "auto_escalate": True}


class Router:
    def __init__(self):
        self._routing = load_routing_config()

    def reload(self):
        self._routing = load_routing_config()

    def route(self, prompt: str) -> tuple[ComplexityTier, ModelConfig]:
        """Classify prompt and return (tier, model_config)."""
        classifier = get_classifier()
        tier = classifier.predict(prompt)
        model_key = self._routing.get(tier, DEFAULT_ROUTING[tier])
        model_config = MODEL_REGISTRY.get(model_key)

        if model_config is None or not model_config.enabled:
            # Fallback to any enabled model
            for key, cfg in MODEL_REGISTRY.items():
                if cfg.enabled:
                    logger.warning(f"Primary model {model_key} unavailable, falling back to {key}")
                    return tier, cfg
            raise RuntimeError("No enabled models available")

        return tier, model_config

    def get_verifier_model(self) -> ModelConfig:
        return MODEL_REGISTRY[VERIFIER_MODEL_KEY]

    def update_mapping(self, tier: ComplexityTier, model_key: str) -> None:
        if model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key: {model_key}")
        self._routing[tier] = model_key
        save_routing_config(self._routing)

    def get_mapping(self) -> dict[str, str]:
        return {tier.name: key for tier, key in self._routing.items()}


_router: Optional[Router] = None


def get_router() -> Router:
    global _router
    if _router is None:
        _router = Router()
    return _router
