#!/usr/bin/env python3
"""
CLI utilities for LLM Cost Autopilot.

Usage:
    python scripts.py train           — Train the complexity classifier
    python scripts.py evaluate        — Evaluate classifier accuracy
    python scripts.py seed-db         — Seed the database with mock data for demo
    python scripts.py load-test N     — Send N diverse prompts through the API
"""

import sys
import json
import asyncio
import random
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def train():
    from app.classifier import get_classifier
    print("Training classifier...")
    clf = get_classifier()
    metrics = clf.train()
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Samples:  {metrics['n_samples']}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")


def evaluate():
    import numpy as np
    from app.classifier import get_classifier, extract_features, ComplexityTier

    path = Path("data/labeled_prompts.json")
    with open(path) as f:
        data = json.load(f)

    clf = get_classifier()
    correct = 0
    per_tier = {1: {"correct": 0, "total": 0}, 2: {"correct": 0, "total": 0}, 3: {"correct": 0, "total": 0}}

    for item in data:
        predicted = clf.predict(item["prompt"])
        actual = item["tier"]
        per_tier[actual]["total"] += 1
        if int(predicted) == actual:
            correct += 1
            per_tier[actual]["correct"] += 1

    print(f"\nOverall accuracy: {correct / len(data):.2%} ({correct}/{len(data)})")
    for tier, counts in per_tier.items():
        tier_names = {1: "Simple  ", 2: "Moderate", 3: "Complex "}
        acc = counts["correct"] / counts["total"] if counts["total"] else 0
        print(f"  Tier {tier} ({tier_names[tier]}): {acc:.2%} ({counts['correct']}/{counts['total']})")


def seed_db():
    """Populate database with realistic mock data for dashboard demo."""
    import sqlite3
    import hashlib
    from app.database import init_db, DB_PATH

    init_db()

    sample_prompts = [
        ("What is the capital of France?", 1, "claude-haiku-4-5-20251001", "anthropic", 0.00002, 120),
        ("Summarize this article about AI safety.", 2, "gpt-4o-mini", "openai", 0.00008, 450),
        ("Write a comprehensive analysis of microservices vs monolith.", 3, "gpt-4o", "openai", 0.0035, 1800),
        ("Fix the typo: 'recieve'", 1, "claude-haiku-4-5-20251001", "anthropic", 0.000015, 90),
        ("Compare REST vs GraphQL with examples.", 2, "gpt-4o-mini", "openai", 0.00012, 620),
        ("Design a distributed rate limiting system.", 3, "gpt-4o", "openai", 0.0042, 2100),
        ("What does API stand for?", 1, "claude-haiku-4-5-20251001", "anthropic", 0.000012, 80),
        ("Classify these customer reviews by sentiment.", 2, "gpt-4o-mini", "openai", 0.00009, 390),
        ("Write a 1500-word persuasive essay on UBI.", 3, "gpt-4o", "openai", 0.0051, 2400),
        ("Translate 'Good morning' to Spanish.", 1, "claude-haiku-4-5-20251001", "anthropic", 0.000018, 110),
    ]

    conn = sqlite3.connect(DB_PATH)
    now = datetime.utcnow()

    inserted = 0
    for i in range(200):
        prompt_data = random.choice(sample_prompts)
        prompt, tier, model, provider, cost, latency = prompt_data
        days_ago = random.randint(0, 7)
        hours_ago = random.randint(0, 23)
        ts = (now - timedelta(days=days_ago, hours=hours_ago)).isoformat()
        quality = random.uniform(3.5, 5.0) if tier > 1 else random.uniform(4.0, 5.0)
        was_escalated = 1 if quality < 4.0 and random.random() < 0.3 else 0

        conn.execute("""
            INSERT INTO requests (
                timestamp, prompt_hash, prompt_preview, complexity_tier,
                routed_model, provider, input_tokens, output_tokens,
                cost_usd, latency_ms, quality_score, was_escalated, error
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            ts,
            hashlib.sha256(f"{prompt}{i}".encode()).hexdigest()[:16],
            prompt[:200],
            tier,
            model,
            provider,
            random.randint(50, 800),
            random.randint(20, 400),
            cost * random.uniform(0.8, 1.2),
            int(latency * random.uniform(0.7, 1.3)),
            round(quality, 1),
            was_escalated,
            None,
        ))
        inserted += 1

    conn.commit()
    conn.close()
    print(f"Seeded {inserted} mock requests into {DB_PATH}")


async def _load_test_async(n: int):
    import httpx

    prompts = [
        "What is the capital of Japan?",
        "Fix the grammar: 'She dont know nothing.'",
        "What does JSON stand for?",
        "Translate 'Hello' to French.",
        "Summarize the pros and cons of remote work.",
        "Compare Docker vs virtual machines.",
        "Classify this review as positive/negative: 'Great product, fast delivery!'",
        "Write a professional email declining a meeting.",
        "What are the SOLID principles in software engineering?",
        "Summarize this paragraph: The industrial revolution transformed...",
        "Write a comprehensive analysis of microservices architecture for a Fortune 500.",
        "Design a distributed rate limiting system with no single point of failure.",
        "Analyze the ethical implications of predictive policing algorithms.",
        "Write a 1000-word essay on the future of artificial intelligence.",
        "Compare REST, GraphQL, and gRPC for a real-time collaborative editing app.",
    ]

    results = {"success": 0, "error": 0, "total_cost": 0.0}
    tier_counts = {1: 0, 2: 0, 3: 0}

    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(n):
            prompt = random.choice(prompts)
            try:
                resp = await client.post(
                    "http://localhost:8000/v1/completions",
                    json={"messages": [{"role": "user", "content": prompt}]},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results["success"] += 1
                    results["total_cost"] += data.get("cost_usd", 0)
                    tier = data.get("complexity_tier", 1)
                    tier_counts[tier] = tier_counts.get(tier, 0) + 1
                    print(f"[{i+1}/{n}] Tier {tier} → {data.get('model_used')} (${data.get('cost_usd', 0):.6f})")
                else:
                    results["error"] += 1
                    print(f"[{i+1}/{n}] Error: {resp.status_code}")
            except Exception as e:
                results["error"] += 1
                print(f"[{i+1}/{n}] Failed: {e}")

    print(f"\n--- Load Test Results ({n} requests) ---")
    print(f"Success: {results['success']}  Errors: {results['error']}")
    print(f"Total cost: ${results['total_cost']:.4f}")
    print(f"Tier distribution: {tier_counts}")


def load_test(n: int):
    asyncio.run(_load_test_async(n))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "train":
        train()
    elif cmd == "evaluate":
        evaluate()
    elif cmd == "seed-db":
        seed_db()
    elif cmd == "load-test":
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        load_test(n)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)
