import hashlib
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.models import MODEL_REGISTRY, ModelConfig
from app.interface import send_request
from app.router import get_router, ComplexityTier
from app.classifier import get_classifier
from app.database import init_db, log_request, get_stats, get_recent_requests, get_gpt4o_baseline_cost, RequestLog
from app.verifier import schedule_verification

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Cost Autopilot",
    description="Intelligent LLM routing layer. Sends each request to the cheapest capable model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    init_db()
    logger.info("LLM Cost Autopilot started")


# ─── Request / Response schemas ─────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class CompletionRequest(BaseModel):
    messages: list[Message]
    max_tokens: int = 1024
    system: Optional[str] = None
    force_tier: Optional[int] = None   # 1, 2, or 3 — override routing


class CompletionResponse(BaseModel):
    output: str
    model_used: str
    provider: str
    complexity_tier: int
    complexity_tier_name: str
    cost_usd: float
    latency_ms: int
    tokens: dict
    request_id: int
    error: Optional[str] = None


class RoutingConfigUpdate(BaseModel):
    tier_1_model: Optional[str] = None   # model registry key
    tier_2_model: Optional[str] = None
    tier_3_model: Optional[str] = None


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.post("/v1/completions", response_model=CompletionResponse)
async def completions(request: CompletionRequest, background_tasks: BackgroundTasks):
    """
    Main endpoint. Routes the request to the optimal model automatically.
    The caller does not choose the model — the router does.
    """
    # Extract the user's latest message as the routing prompt
    user_messages = [m for m in request.messages if m.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="At least one user message required")

    prompt = user_messages[-1].content
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]

    router = get_router()

    # Classify and route
    if request.force_tier:
        try:
            tier = ComplexityTier(request.force_tier)
            _, model_config = router.route(prompt)
            # Override with forced tier's model
            from app.router import DEFAULT_ROUTING, load_routing_config
            mapping = load_routing_config()
            from app.models import MODEL_REGISTRY
            model_key = mapping.get(tier)
            model_config = MODEL_REGISTRY.get(model_key, model_config)
        except (ValueError, Exception):
            tier, model_config = router.route(prompt)
    else:
        tier, model_config = router.route(prompt)

    logger.info(f"Routing '{prompt[:60]}...' → Tier {tier} → {model_config.display_name}")

    # Call the model
    response = await send_request(
        prompt=prompt,
        model_config=model_config,
        system_prompt=request.system,
        max_tokens=request.max_tokens,
    )

    # Log to database
    log = RequestLog(
        id=None,
        timestamp=datetime.utcnow().isoformat(),
        prompt_hash=prompt_hash,
        prompt_preview=prompt[:200],
        complexity_tier=int(tier),
        routed_model=model_config.model_id,
        provider=model_config.provider.value,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
        quality_score=None,
        was_escalated=False,
        escalated_model=None,
        escalation_cost_delta=None,
        error=response.error,
    )
    request_id = log_request(log)

    # Schedule async quality verification (non-blocking)
    if not response.error:
        background_tasks.add_task(
            schedule_verification,
            prompt,
            response,
            request_id,
            tier,
        )

    tier_names = {1: "simple", 2: "moderate", 3: "complex"}

    return CompletionResponse(
        output=response.text,
        model_used=model_config.model_id,
        provider=model_config.provider.value,
        complexity_tier=int(tier),
        complexity_tier_name=tier_names.get(int(tier), "unknown"),
        cost_usd=response.cost_usd,
        latency_ms=response.latency_ms,
        tokens={"input": response.input_tokens, "output": response.output_tokens},
        request_id=request_id,
        error=response.error,
    )


@app.get("/v1/models")
async def list_models():
    """List all available models, their costs, and current routing assignment."""
    router = get_router()
    mapping = router.get_mapping()

    models = []
    for key, config in MODEL_REGISTRY.items():
        models.append({
            "key": key,
            "display_name": config.display_name,
            "provider": config.provider.value,
            "model_id": config.model_id,
            "cost_per_1k_input_tokens": config.cost_per_input_token * 1000,
            "cost_per_1k_output_tokens": config.cost_per_output_token * 1000,
            "quality_tier": config.quality_tier.value,
            "avg_latency_ms": config.average_latency_ms,
            "enabled": config.enabled,
            "currently_assigned_tiers": [
                tier for tier, mk in mapping.items() if mk == key
            ],
        })

    return {"models": models, "routing": mapping}


@app.get("/v1/stats")
async def get_statistics(days: int = 7):
    """Cost savings summary and routing statistics."""
    stats = get_stats(days=days)
    baseline = get_gpt4o_baseline_cost(days=days)
    actual_cost = stats.get("total_cost") or 0.0

    savings = baseline - actual_cost
    savings_pct = (savings / baseline * 100) if baseline > 0 else 0.0

    return {
        "period_days": days,
        "total_requests": stats.get("total_requests", 0),
        "actual_cost_usd": round(actual_cost, 6),
        "baseline_cost_usd": round(baseline, 6),
        "savings_usd": round(savings, 6),
        "savings_percent": round(savings_pct, 2),
        "avg_cost_per_request_usd": round(stats.get("avg_cost") or 0, 8),
        "avg_latency_ms": round(stats.get("avg_latency") or 0, 1),
        "avg_quality_score": round(stats.get("avg_quality") or 0, 2),
        "escalations": stats.get("escalations", 0),
        "escalation_rate_percent": round(
            (stats.get("escalations", 0) / max(stats.get("total_requests", 1), 1)) * 100, 2
        ),
        "by_model": stats.get("by_model", []),
        "by_tier": stats.get("by_tier", []),
        "daily": stats.get("daily", []),
    }


@app.get("/v1/requests")
async def recent_requests(limit: int = 50):
    """Recent request log for inspection."""
    return {"requests": get_recent_requests(limit=limit)}


@app.put("/v1/routing-config")
async def update_routing_config(update: RoutingConfigUpdate):
    """Update tier-to-model mapping without redeploying."""
    router = get_router()
    updated = []

    tier_map = {
        "tier_1_model": ComplexityTier.SIMPLE,
        "tier_2_model": ComplexityTier.MODERATE,
        "tier_3_model": ComplexityTier.COMPLEX,
    }

    for field, tier in tier_map.items():
        model_key = getattr(update, field)
        if model_key:
            if model_key not in MODEL_REGISTRY:
                raise HTTPException(status_code=400, detail=f"Unknown model key: {model_key}. Valid keys: {list(MODEL_REGISTRY.keys())}")
            router.update_mapping(tier, model_key)
            updated.append(f"{tier.name} → {model_key}")

    router.reload()
    return {"updated": updated, "current_routing": router.get_mapping()}


@app.post("/v1/classifier/train")
async def train_classifier():
    """Manually trigger classifier retraining."""
    classifier = get_classifier()
    try:
        metrics = classifier.train()
        return {"status": "trained", "metrics": metrics}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ImportError as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "ok", "service": "llm-cost-autopilot"}
