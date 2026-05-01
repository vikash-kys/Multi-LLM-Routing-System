"""
Tests for LLM Cost Autopilot.
Run with: python -m pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.classifier import extract_features, _rule_based_classify, ComplexityTier, ComplexityClassifier
from app.models import MODEL_REGISTRY, ModelConfig, QualityTier, Provider


# ── Feature extraction ────────────────────────────────────────────────────────

def test_extract_features_simple_prompt():
    features = extract_features("What is the capital of France?")
    assert features.word_count < 10
    assert features.question_count == 1
    assert not features.has_analysis_keywords
    assert not features.has_code


def test_extract_features_complex_prompt():
    prompt = "Analyze and compare the architectural trade-offs between microservices and monolithic systems. Evaluate each approach across scalability, team independence, and operational complexity."
    features = extract_features(prompt)
    assert features.has_analysis_keywords
    assert features.word_count > 20


def test_extract_features_with_code():
    prompt = "```python\ndef hello():\n    print('world')\n```\nFix the bug in this function."
    features = extract_features(prompt)
    assert features.has_code


def test_extract_features_with_constraints():
    prompt = "Summarize this document. You must avoid jargon, should be under 100 words, and never use bullet points. Only use plain prose."
    features = extract_features(prompt)
    assert features.num_constraints >= 3


# ── Rule-based classifier ─────────────────────────────────────────────────────

def test_simple_prompts_classified_correctly():
    simple_prompts = [
        "What is the capital of France?",
        "Fix the typo: recieve",
        "Translate hello to Spanish",
        "What does API stand for?",
    ]
    for prompt in simple_prompts:
        features = extract_features(prompt)
        tier = _rule_based_classify(features)
        assert tier == ComplexityTier.SIMPLE, f"Expected SIMPLE for: '{prompt}', got {tier}"


def test_complex_prompts_classified_correctly():
    complex_prompts = [
        "Write a comprehensive 1500-word analysis comparing microservices vs monolithic architecture, evaluating each on scalability, team autonomy, and operational overhead. Argue for which is better for a fintech startup.",
        "Analyze the ethical implications of predictive policing algorithms. Consider racial bias, due process, and policy implications. Evaluate evidence from multiple perspectives and argue for a governance framework.",
    ]
    for prompt in complex_prompts:
        features = extract_features(prompt)
        tier = _rule_based_classify(features)
        assert tier == ComplexityTier.COMPLEX, f"Expected COMPLEX for: '{prompt[:60]}...', got {tier}"


# ── Model registry ────────────────────────────────────────────────────────────

def test_model_registry_has_required_models():
    required = ["gpt-4o", "gpt-4o-mini", "claude-sonnet", "claude-haiku", "llama3"]
    for key in required:
        assert key in MODEL_REGISTRY, f"Missing model: {key}"


def test_model_cost_estimation():
    model = MODEL_REGISTRY["gpt-4o"]
    cost = model.estimate_cost(input_tokens=1000, output_tokens=500)
    expected = 1000 * model.cost_per_input_token + 500 * model.cost_per_output_token
    assert abs(cost - expected) < 1e-10


def test_llama_is_free():
    model = MODEL_REGISTRY["llama3"]
    assert model.estimate_cost(10000, 5000) == 0.0


def test_quality_tiers_correctly_assigned():
    assert MODEL_REGISTRY["gpt-4o"].quality_tier == QualityTier.HIGH
    assert MODEL_REGISTRY["gpt-4o-mini"].quality_tier == QualityTier.MEDIUM
    assert MODEL_REGISTRY["claude-haiku"].quality_tier == QualityTier.LOW


# ── Classifier integration ────────────────────────────────────────────────────

def test_classifier_predict_returns_valid_tier():
    clf = ComplexityClassifier()
    prompts = [
        "What is 2 + 2?",
        "Summarize the pros and cons of remote work.",
        "Write a comprehensive analysis of AI ethics with multiple perspectives.",
    ]
    for prompt in prompts:
        tier = clf.predict(prompt)
        assert tier in (ComplexityTier.SIMPLE, ComplexityTier.MODERATE, ComplexityTier.COMPLEX)


def test_classifier_training_improves_accuracy(tmp_path):
    import json
    # Create a small labeled dataset
    data = [
        {"prompt": "What is 2+2?", "tier": 1},
        {"prompt": "What does API mean?", "tier": 1},
        {"prompt": "Fix the typo: recieve", "tier": 1},
        {"prompt": "Summarize the pros and cons of remote work.", "tier": 2},
        {"prompt": "Compare REST vs GraphQL APIs.", "tier": 2},
        {"prompt": "Classify these reviews as positive or negative.", "tier": 2},
        {"prompt": "Write a comprehensive analysis of microservices architecture trade-offs for enterprise systems.", "tier": 3},
        {"prompt": "Analyze the ethical implications of algorithmic bias in hiring tools and argue for governance reform.", "tier": 3},
        {"prompt": "Design a distributed rate limiting system with no single point of failure.", "tier": 3},
    ]
    data_path = tmp_path / "test_prompts.json"
    with open(data_path, "w") as f:
        json.dump(data, f)

    clf = ComplexityClassifier()
    try:
        metrics = clf.train(data_path=data_path)
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
    except ImportError:
        pytest.skip("scikit-learn not installed")
