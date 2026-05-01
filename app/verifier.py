import asyncio
import logging
import hashlib
from typing import Optional

from app.models import MODEL_REGISTRY, Response
from app.interface import send_request
from app.router import get_router, get_quality_config
from app.classifier import get_classifier, ComplexityTier

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a quality evaluation assistant. You will be shown a user prompt
and two responses: Response A (from a cheaper model) and Response B (from a high-quality model).

Rate how well Response A matches the quality and correctness of Response B on a scale of 1 to 5:
- 5: Identical or equivalent quality; Response A is fully correct and complete
- 4: Very close; minor differences that don't affect usefulness
- 3: Acceptable but noticeably inferior; key points present but some gaps
- 2: Significantly worse; important content missing or incorrect
- 1: Completely wrong or unhelpful compared to Response B

Respond ONLY with a single integer: 1, 2, 3, 4, or 5. Nothing else."""


async def verify_response(
    prompt: str,
    cheap_response: Response,
    request_id: int,
    complexity_tier: ComplexityTier,
) -> None:
    """
    Async quality verification. Runs after the user has already received their response.
    Compares cheap model output against the top-tier model using LLM-as-judge.
    """
    from app.database import update_quality_score

    router = get_router()
    quality_cfg = get_quality_config()
    min_score = quality_cfg.get("min_judge_score", 4)
    auto_escalate = quality_cfg.get("auto_escalate", True)

    verifier_model = router.get_verifier_model()

    # Skip verification if cheap model IS the verifier model
    if cheap_response.model_id == verifier_model.model_id:
        update_quality_score(request_id, 5.0, False)
        return

    try:
        # Get ground-truth response from the best model
        reference_response = await send_request(
            prompt=prompt,
            model_config=verifier_model,
            max_tokens=1024,
        )

        if reference_response.error:
            logger.warning(f"Verifier model failed: {reference_response.error}")
            return

        # LLM-as-judge: score the cheap model's output
        judge_prompt = f"""User Prompt:
{prompt}

---
Response A (to evaluate):
{cheap_response.text}

---
Response B (reference):
{reference_response.text}

Rate Response A compared to Response B (1-5):"""

        # Use gpt-4o-mini as the judge (cheap but capable for scoring)
        judge_model = MODEL_REGISTRY.get("gpt-4o-mini") or verifier_model
        judge_response = await send_request(
            prompt=judge_prompt,
            model_config=judge_model,
            system_prompt=JUDGE_SYSTEM_PROMPT,
            max_tokens=5,
        )

        score = _parse_score(judge_response.text)
        logger.info(f"Quality score for request {request_id}: {score}/5")

        was_escalated = False
        escalated_model = None
        cost_delta = None

        if score < min_score:
            logger.warning(
                f"Low quality score ({score}) for request {request_id}. "
                f"Cheap model: {cheap_response.model_id}, should have used {verifier_model.model_id}"
            )
            # Feed failure back to classifier
            correct_tier = _infer_correct_tier(complexity_tier)
            get_classifier().add_failure(prompt, correct_tier)

            if auto_escalate:
                was_escalated = True
                escalated_model = verifier_model.model_id
                cost_delta = reference_response.cost_usd - cheap_response.cost_usd
                logger.info(f"Auto-escalation logged. Cost delta: ${cost_delta:.6f}")

        update_quality_score(
            request_id=request_id,
            score=score,
            was_escalated=was_escalated,
            escalated_model=escalated_model,
            cost_delta=cost_delta,
        )

    except Exception as e:
        logger.error(f"Verification failed for request {request_id}: {e}")


def _parse_score(text: str) -> float:
    """Extract numeric score 1-5 from judge response."""
    text = text.strip()
    for char in text:
        if char.isdigit():
            val = int(char)
            if 1 <= val <= 5:
                return float(val)
    return 3.0  # Default to middle if unparseable


def _infer_correct_tier(current_tier: ComplexityTier) -> ComplexityTier:
    """If routing failed, the request needed at least one tier higher."""
    if current_tier == ComplexityTier.SIMPLE:
        return ComplexityTier.MODERATE
    return ComplexityTier.COMPLEX


def schedule_verification(
    prompt: str,
    cheap_response: Response,
    request_id: int,
    complexity_tier: ComplexityTier,
) -> None:
    """Schedule verification as a background asyncio task."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(
                verify_response(prompt, cheap_response, request_id, complexity_tier)
            )
        else:
            logger.warning("No running event loop for background verification")
    except Exception as e:
        logger.error(f"Could not schedule verification: {e}")
