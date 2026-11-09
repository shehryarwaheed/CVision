import numpy as np
import pandas as pd

REQUIRED_QUALITY_FEATURES = [
    "quantified_sentence_ratio",
    "section_count",
    "action_verbs_per_100_words",
    "lexical_diversity",
]

# Calibrated targets grounded near the 75th percentile of the training distribution
TARGET_QUANTIFIED_RATIO = 0.25      # 25% of sentences contain non-year metric numbers
TARGET_SECTION_COUNT = 5.0          # Experience, Education, Skills, Summary, Projects
TARGET_ACTION_VERB_DENSITY = 3.5    # 3.5 action verbs per 100 words
TARGET_LEXICAL_DIVERSITY = 0.50     # 50% unique tokens across the first 300 tokens


def compute_cv_quality_score(df: pd.DataFrame) -> np.ndarray:
    """
    Computes transparent 0-100 heuristic quality score from DataFrame of engineered features.
    Raises ValueError if required feature columns are missing.
    """
    missing = [c for c in REQUIRED_QUALITY_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required structural feature columns for quality scoring: {missing}")

    q_ratio = df["quantified_sentence_ratio"].to_numpy().astype(float)
    sections = df["section_count"].to_numpy().astype(float)
    verbs = df["action_verbs_per_100_words"].to_numpy().astype(float)
    lex = df["lexical_diversity"].to_numpy().astype(float)

    # 1. Quantified impact component (0-30 pts)
    impact_score = np.clip((q_ratio / TARGET_QUANTIFIED_RATIO) * 30.0, 0.0, 30.0)

    # 2. Section coverage component (0-30 pts)
    section_score = np.clip((sections / TARGET_SECTION_COUNT) * 30.0, 0.0, 30.0)

    # 3. Action verb density component (0-20 pts)
    verb_score = np.clip((verbs / TARGET_ACTION_VERB_DENSITY) * 20.0, 0.0, 20.0)

    # 4. Lexical diversity component (0-20 pts)
    diversity_score = np.clip((lex / TARGET_LEXICAL_DIVERSITY) * 20.0, 0.0, 20.0)

    total_score = np.round(impact_score + section_score + verb_score + diversity_score, 2)
    return total_score


if __name__ == "__main__":
    # Self-test unit verification
    toy_data = pd.DataFrame({
        "quantified_sentence_ratio": [0.0, 0.12, 0.25, 0.50],
        "section_count":            [1.0, 3.0,  5.0,  7.0],
        "action_verbs_per_100_words": [0.5, 2.0,  3.5,  6.0],
        "lexical_diversity":         [0.20, 0.35, 0.50, 0.70]
    })

    scores = compute_cv_quality_score(toy_data)
    print("CVision Quality Scorer Self-Test:")
    for idx, s in enumerate(scores):
        print(f"  Candidate {idx + 1}: Score = {s:.2f} / 100.0")

    # Invariants verification
    assert (scores >= 0.0).all() and (scores <= 100.0).all(), "Scores must be strictly bounded in [0, 100]!"
    assert np.all(np.diff(scores) > 0), "Scores must be monotonically increasing with better metrics!"
    print("Self-test passed successfully. All bounds and monotonicity verified.")