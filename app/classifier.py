import json
import re
import logging
import pickle
from pathlib import Path
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

MODEL_PATH = Path("data/classifier.pkl")
TRAINING_DATA_PATH = Path("data/labeled_prompts.json")


class ComplexityTier(IntEnum):
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3


# Keywords that signal higher complexity
ANALYSIS_KEYWORDS = [
    "analyze", "analyse", "compare", "contrast", "evaluate", "assess",
    "critique", "argue", "debate", "justify", "synthesize", "interpret",
    "infer", "examine", "investigate", "explore", "discuss", "elaborate",
]

MULTI_FACET_PHRASES = [
    "multiple perspectives", "multiple angles", "trade-offs", "trade offs",
    "pros and cons", "for and against", "implications", "governance",
    "comprehensive", "in-depth", "detailed analysis", "nuanced",
    "across multiple", "consider", "evaluate evidence",
]

CREATIVE_KEYWORDS = [
    "write", "create", "generate", "draft", "compose", "brainstorm",
    "imagine", "design", "invent", "story", "poem", "essay",
]

SIMPLE_KEYWORDS = [
    "extract", "list", "summarize", "translate", "reformat", "convert",
    "find", "identify", "what is", "define", "spell", "fix",
]


@dataclass
class PromptFeatures:
    token_count: int
    sentence_count: int
    has_analysis_keywords: bool
    has_creative_keywords: bool
    has_simple_keywords: bool
    num_constraints: int
    has_context_provided: bool
    output_format_complexity: int   # 0=none, 1=simple, 2=structured, 3=complex
    question_count: int
    has_code: bool
    has_multi_facet_phrases: bool
    has_numbers_or_data: bool
    word_count: int

    def to_array(self) -> np.ndarray:
        return np.array([
            self.token_count,
            self.sentence_count,
            int(self.has_analysis_keywords),
            int(self.has_creative_keywords),
            int(self.has_simple_keywords),
            self.num_constraints,
            int(self.has_context_provided),
            self.output_format_complexity,
            self.question_count,
            int(self.has_code),
            int(self.has_numbers_or_data),
            self.word_count,
            int(self.has_multi_facet_phrases),
        ], dtype=float)


def extract_features(prompt: str) -> PromptFeatures:
    lower = prompt.lower()
    words = prompt.split()
    word_count = len(words)
    token_count = word_count + len(prompt) // 4  # rough estimate
    sentences = re.split(r'[.!?]+', prompt)
    sentence_count = len([s for s in sentences if s.strip()])

    has_analysis_keywords = any(kw in lower for kw in ANALYSIS_KEYWORDS)
    has_creative_keywords = any(kw in lower for kw in CREATIVE_KEYWORDS)
    has_simple_keywords = any(kw in lower for kw in SIMPLE_KEYWORDS)

    # Constraint signals: "must", "should", "do not", "only", "without", "limit"
    constraint_words = ["must", "should", "do not", "don't", "only", "without",
                        "limit", "ensure", "avoid", "never", "always", "exactly"]
    num_constraints = sum(1 for c in constraint_words if c in lower)

    # Context provided: long prompt with colons or quoted material
    has_context_provided = (
        len(prompt) > 500
        or ":" in prompt
        or bool(re.search(r'"""|\'\'\''  , prompt))
        or "context:" in lower
        or "background:" in lower
    )

    # Output format complexity
    format_keywords_simple = ["list", "bullet", "yes or no", "true or false", "one word"]
    format_keywords_structured = ["json", "table", "csv", "structured", "format", "markdown"]
    format_keywords_complex = ["report", "essay", "detailed", "comprehensive", "in-depth"]

    if any(kw in lower for kw in format_keywords_complex):
        output_format_complexity = 3
    elif any(kw in lower for kw in format_keywords_structured):
        output_format_complexity = 2
    elif any(kw in lower for kw in format_keywords_simple):
        output_format_complexity = 1
    else:
        output_format_complexity = 0

    question_count = prompt.count("?")
    has_code = bool(re.search(r'```|def |class |import |function |var |const |let ', prompt))
    has_numbers_or_data = bool(re.search(r'\d+\.?\d*%|\$\d|\d{4}|\bdata\b|\bstatistic', lower))
    has_multi_facet_phrases = any(phrase in lower for phrase in MULTI_FACET_PHRASES)

    return PromptFeatures(
        token_count=token_count,
        sentence_count=sentence_count,
        has_analysis_keywords=has_analysis_keywords,
        has_creative_keywords=has_creative_keywords,
        has_simple_keywords=has_simple_keywords,
        num_constraints=num_constraints,
        has_context_provided=has_context_provided,
        output_format_complexity=output_format_complexity,
        question_count=question_count,
        has_code=has_code,
        has_numbers_or_data=has_numbers_or_data,
        has_multi_facet_phrases=has_multi_facet_phrases,
        word_count=word_count,
    )


def _rule_based_classify(features: PromptFeatures) -> ComplexityTier:
    """Fallback rule-based classifier when no trained model exists."""
    score = 0

    if features.token_count > 500:
        score += 2
    elif features.token_count > 150:
        score += 1

    if features.has_analysis_keywords:
        score += 3
    if features.has_creative_keywords:
        score += 1
    if features.has_simple_keywords:
        score -= 1
    if features.has_multi_facet_phrases:
        score += 2

    score += min(features.num_constraints, 3)
    score += features.output_format_complexity
    score += features.question_count

    if features.has_code:
        score += 1
    if features.has_numbers_or_data:
        score += 1

    if score <= 2:
        return ComplexityTier.SIMPLE
    elif score <= 4:
        return ComplexityTier.MODERATE
    else:
        return ComplexityTier.COMPLEX


class ComplexityClassifier:
    def __init__(self):
        self._model = None
        self._load()

    def _load(self):
        if MODEL_PATH.exists():
            try:
                with open(MODEL_PATH, "rb") as f:
                    self._model = pickle.load(f)
                logger.info("Loaded trained classifier from disk")
            except Exception as e:
                logger.warning(f"Could not load classifier: {e}. Using rule-based fallback.")
                self._model = None

    def predict(self, prompt: str) -> ComplexityTier:
        features = extract_features(prompt)
        if self._model is not None:
            try:
                X = features.to_array().reshape(1, -1)
                label = int(self._model.predict(X)[0])
                return ComplexityTier(label)
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}. Falling back to rules.")
        return _rule_based_classify(features)

    def train(self, data_path: Optional[Path] = None) -> dict:
        """Train on labeled data. Returns metrics dict."""
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score, confusion_matrix
        except ImportError:
            raise ImportError("scikit-learn required: pip install scikit-learn")

        path = data_path or TRAINING_DATA_PATH
        if not path.exists():
            raise FileNotFoundError(f"Training data not found: {path}")

        with open(path) as f:
            data = json.load(f)

        X, y = [], []
        for item in data:
            features = extract_features(item["prompt"])
            X.append(features.to_array())
            y.append(int(item["tier"]))

        X = np.array(X)
        y = np.array(y)

        stratify = y if len(y) >= 15 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )

        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred).tolist()

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(clf, f)

        self._model = clf
        logger.info(f"Trained classifier. Accuracy: {acc:.2%}")
        return {"accuracy": acc, "confusion_matrix": cm, "n_samples": len(X)}

    def add_failure(self, prompt: str, correct_tier: ComplexityTier):
        """Add a routing failure as a new training example."""
        path = TRAINING_DATA_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        if path.exists():
            with open(path) as f:
                data = json.load(f)

        data.append({"prompt": prompt, "tier": int(correct_tier), "source": "routing_failure"})

        with open(path, "w") as f:
            json.dump(data, f, indent=2)


# Module-level singleton
_classifier: Optional[ComplexityClassifier] = None


def get_classifier() -> ComplexityClassifier:
    global _classifier
    if _classifier is None:
        _classifier = ComplexityClassifier()
    return _classifier
