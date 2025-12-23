"""
Linear probe training and evaluation for Phase 2.

Two probe types:
- Reality probe: predicts tool_used (did the action actually happen?)
- Narrative probe: predicts claims_action (does the text claim action?)

Key evaluation:
- On fake_escalation episodes: reality probe should say "no tool"
  even though narrative says "claims action"
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# =============================================================================
# PROBE TRAINING
# =============================================================================

def train_reality_probe(
    X: np.ndarray,
    y_tool_used: np.ndarray,
    balance_classes: bool = True,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a linear probe to predict tool_used from activations.

    This is the "reality" probe - does the model internally represent
    whether it actually called the tool?

    Args:
        X: Activation vectors [n_samples, hidden_dim]
        y_tool_used: Ground truth tool usage [n_samples]
        balance_classes: Use class weights to handle imbalance
        random_state: Random seed

    Returns:
        Trained LogisticRegression probe
    """
    probe = LogisticRegression(
        class_weight="balanced" if balance_classes else None,
        max_iter=1000,
        random_state=random_state,
    )
    probe.fit(X, y_tool_used)
    return probe


def train_narrative_probe(
    X: np.ndarray,
    y_claims_action: np.ndarray,
    balance_classes: bool = True,
    random_state: int = 42,
) -> LogisticRegression:
    """
    Train a linear probe to predict claims_action from activations.

    This is the "narrative" probe - does the model internally represent
    what it's claiming to do?

    Args:
        X: Activation vectors [n_samples, hidden_dim]
        y_claims_action: Ground truth claim labels [n_samples]
        balance_classes: Use class weights to handle imbalance
        random_state: Random seed

    Returns:
        Trained LogisticRegression probe
    """
    probe = LogisticRegression(
        class_weight="balanced" if balance_classes else None,
        max_iter=1000,
        random_state=random_state,
    )
    probe.fit(X, y_claims_action)
    return probe


# =============================================================================
# EVALUATION: OVERALL
# =============================================================================

@dataclass
class ProbeMetrics:
    """Metrics for probe evaluation."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    n_samples: int


def evaluate_probe(
    probe: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
) -> ProbeMetrics:
    """
    Evaluate probe on test data.

    Returns:
        ProbeMetrics with accuracy, precision, recall, F1
    """
    from sklearn.metrics import precision_recall_fscore_support

    y_pred = probe.predict(X)
    acc = accuracy_score(y, y_pred)

    # Handle binary classification
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average="binary", zero_division=0
    )

    cm = confusion_matrix(y, y_pred)

    return ProbeMetrics(
        accuracy=acc,
        precision=precision,
        recall=recall,
        f1=f1,
        confusion=cm,
        n_samples=len(y),
    )


def cross_validate_probe(
    X: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    balance_classes: bool = True,
) -> dict:
    """
    Cross-validate probe performance.

    Returns:
        Dict with mean/std accuracy across folds
    """
    probe = LogisticRegression(
        class_weight="balanced" if balance_classes else None,
        max_iter=1000,
    )

    scores = cross_val_score(probe, X, y, cv=n_folds, scoring="accuracy")

    return {
        "mean_accuracy": scores.mean(),
        "std_accuracy": scores.std(),
        "fold_scores": scores,
        "n_folds": n_folds,
    }


# =============================================================================
# EVALUATION: PER-CATEGORY
# =============================================================================

@dataclass
class CategoryMetrics:
    """Metrics for a single category."""
    category: str
    n_samples: int
    accuracy: float
    mean_prob_true: float  # Mean P(tool_used=True) or P(claims=True)
    std_prob_true: float


def evaluate_by_category(
    probe: LogisticRegression,
    X: np.ndarray,
    y: np.ndarray,
    categories: np.ndarray,
) -> list[CategoryMetrics]:
    """
    Evaluate probe performance broken down by category.

    Args:
        probe: Trained probe
        X: Activation vectors
        y: Ground truth labels
        categories: Category labels for each sample

    Returns:
        List of CategoryMetrics, one per unique category
    """
    results = []
    unique_cats = np.unique(categories)

    for cat in unique_cats:
        mask = categories == cat
        X_cat = X[mask]
        y_cat = y[mask]

        if len(y_cat) == 0:
            continue

        y_pred = probe.predict(X_cat)
        probs = probe.predict_proba(X_cat)[:, 1]  # P(positive class)

        results.append(CategoryMetrics(
            category=cat,
            n_samples=len(y_cat),
            accuracy=accuracy_score(y_cat, y_pred),
            mean_prob_true=probs.mean(),
            std_prob_true=probs.std(),
        ))

    return results


def print_category_metrics(metrics: list[CategoryMetrics], probe_name: str = "Probe"):
    """Pretty print category metrics."""
    print(f"\n{'='*60}")
    print(f"{probe_name} Performance by Category")
    print(f"{'='*60}")
    print(f"{'Category':<25} {'N':>6} {'Acc':>8} {'P(True)':>10}")
    print(f"{'-'*60}")

    for m in sorted(metrics, key=lambda x: x.category):
        print(f"{m.category:<25} {m.n_samples:>6} {m.accuracy:>8.1%} {m.mean_prob_true:>10.3f}")


# =============================================================================
# EVALUATION: FAKE ESCALATION DEEP DIVE
# =============================================================================

@dataclass
class FakeEscalationAnalysis:
    """Analysis of probe behavior on fake escalation episodes."""
    n_fake: int
    mean_prob_tool_used: float
    std_prob_tool_used: float
    probe_says_no_tool_rate: float  # % where probe predicts tool_used=False
    probe_aligned_with_reality: float  # Should be high (probe knows truth)
    probe_aligned_with_narrative: float  # Should be low (probe not fooled)


def analyze_fake_escalations(
    reality_probe: LogisticRegression,
    narrative_probe: Optional[LogisticRegression],
    X: np.ndarray,
    y_tool: np.ndarray,
    y_claims: np.ndarray,
    categories: np.ndarray,
) -> FakeEscalationAnalysis:
    """
    Deep analysis of probe behavior on fake_escalation episodes.

    For fake escalations:
    - Ground truth: tool_used=False, claims_action=True
    - Reality probe should predict: tool_used=False (aligned with reality)
    - If it predicts True, it's being "fooled" by the narrative

    Args:
        reality_probe: Probe trained on tool_used
        narrative_probe: Optional probe trained on claims_action
        X: Activations
        y_tool: Ground truth tool_used
        y_claims: Ground truth claims_action
        categories: Category labels

    Returns:
        FakeEscalationAnalysis with detailed metrics
    """
    fake_mask = categories == "fake_escalation"
    X_fake = X[fake_mask]

    if len(X_fake) == 0:
        return FakeEscalationAnalysis(
            n_fake=0,
            mean_prob_tool_used=0,
            std_prob_tool_used=0,
            probe_says_no_tool_rate=0,
            probe_aligned_with_reality=0,
            probe_aligned_with_narrative=0,
        )

    # Reality probe predictions on fake episodes
    probs_tool = reality_probe.predict_proba(X_fake)[:, 1]  # P(tool_used=True)
    preds_tool = reality_probe.predict(X_fake)

    # For fake episodes, ground truth is tool_used=False
    # So probe_says_no_tool means pred=False, which is CORRECT
    probe_says_no_tool = (preds_tool == False).mean()

    # Aligned with reality = predicts False (correct)
    aligned_reality = probe_says_no_tool

    # Aligned with narrative = predicts True (fooled by the lie)
    aligned_narrative = (preds_tool == True).mean()

    return FakeEscalationAnalysis(
        n_fake=len(X_fake),
        mean_prob_tool_used=probs_tool.mean(),
        std_prob_tool_used=probs_tool.std(),
        probe_says_no_tool_rate=probe_says_no_tool,
        probe_aligned_with_reality=aligned_reality,
        probe_aligned_with_narrative=aligned_narrative,
    )


def print_fake_analysis(analysis: FakeEscalationAnalysis):
    """Pretty print fake escalation analysis."""
    print(f"\n{'='*60}")
    print("FAKE ESCALATION ANALYSIS")
    print("(Episodes where tool_used=False but claims_action=True)")
    print(f"{'='*60}")

    if analysis.n_fake == 0:
        print("No fake escalation episodes found.")
        return

    print(f"N fake episodes: {analysis.n_fake}")
    print(f"\nReality Probe on Fake Episodes:")
    print(f"  Mean P(tool_used=True): {analysis.mean_prob_tool_used:.3f} ± {analysis.std_prob_tool_used:.3f}")
    print(f"  Probe predicts 'no tool': {analysis.probe_says_no_tool_rate:.1%}")
    print(f"\nAlignment:")
    print(f"  Aligned with REALITY (correct):    {analysis.probe_aligned_with_reality:.1%}")
    print(f"  Aligned with NARRATIVE (fooled):   {analysis.probe_aligned_with_narrative:.1%}")

    if analysis.probe_aligned_with_reality > 0.7:
        print(f"\n✓ Probe appears to 'know' the model didn't call the tool!")
    elif analysis.probe_aligned_with_narrative > 0.7:
        print(f"\n✗ Probe is fooled by the narrative (follows the lie)")
    else:
        print(f"\n? Mixed results - probe is uncertain")


# =============================================================================
# CROSS-DOMAIN TRANSFER
# =============================================================================

@dataclass
class TransferResult:
    """Result of cross-domain transfer test."""
    train_domain: str
    test_domain: str
    train_accuracy: float
    test_accuracy: float
    transfer_gap: float  # train_acc - test_acc
    baseline_accuracy: float  # Majority class baseline
    above_chance: bool  # Is test_accuracy > baseline?


def evaluate_transfer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    train_domain: str = "domain_A",
    test_domain: str = "domain_B",
) -> TransferResult:
    """
    Test cross-domain transfer of probe.

    Train probe on domain A, test on domain B.
    If accuracy transfers, suggests shared representation.

    Args:
        X_train, y_train: Training data from domain A
        X_test, y_test: Test data from domain B
        train_domain, test_domain: Names for reporting

    Returns:
        TransferResult with train/test accuracy and transfer gap
    """
    # Train probe on domain A
    probe = train_reality_probe(X_train, y_train)

    # Evaluate on both domains
    train_acc = accuracy_score(y_train, probe.predict(X_train))
    test_acc = accuracy_score(y_test, probe.predict(X_test))

    # Baseline: majority class in test set
    majority_class = np.bincount(y_test.astype(int)).argmax()
    baseline = (y_test == majority_class).mean()

    return TransferResult(
        train_domain=train_domain,
        test_domain=test_domain,
        train_accuracy=train_acc,
        test_accuracy=test_acc,
        transfer_gap=train_acc - test_acc,
        baseline_accuracy=baseline,
        above_chance=test_acc > baseline + 0.05,  # 5% margin
    )


def print_transfer_result(result: TransferResult):
    """Pretty print transfer result."""
    print(f"\n{'='*60}")
    print("CROSS-DOMAIN TRANSFER TEST")
    print(f"{'='*60}")
    print(f"Train domain: {result.train_domain}")
    print(f"Test domain:  {result.test_domain}")
    print(f"\nAccuracy:")
    print(f"  Train: {result.train_accuracy:.1%}")
    print(f"  Test:  {result.test_accuracy:.1%}")
    print(f"  Gap:   {result.transfer_gap:+.1%}")
    print(f"\nBaseline (majority class): {result.baseline_accuracy:.1%}")
    print(f"Above chance: {'YES' if result.above_chance else 'NO'}")

    if result.above_chance and result.transfer_gap < 0.15:
        print(f"\n✓ TRANSFER SUCCESS: Probe generalizes across domains!")
        print(f"  → Suggests shared 'action grounding' representation")
    elif result.above_chance:
        print(f"\n~ PARTIAL TRANSFER: Some generalization, significant gap")
    else:
        print(f"\n✗ NO TRANSFER: Probe doesn't generalize")
        print(f"  → Action grounding may be domain-specific")


# =============================================================================
# FULL EVALUATION PIPELINE
# =============================================================================

def full_probe_evaluation(
    X_train: np.ndarray,
    y_tool_train: np.ndarray,
    y_claims_train: np.ndarray,
    categories_train: np.ndarray,
    X_test: np.ndarray,
    y_tool_test: np.ndarray,
    y_claims_test: np.ndarray,
    categories_test: np.ndarray,
    verbose: bool = True,
) -> dict:
    """
    Run full probe evaluation pipeline.

    Returns dict with all metrics.
    """
    results = {}

    # Train probes
    reality_probe = train_reality_probe(X_train, y_tool_train)
    narrative_probe = train_narrative_probe(X_train, y_claims_train)

    # Overall metrics
    results["reality_overall"] = evaluate_probe(reality_probe, X_test, y_tool_test)
    results["narrative_overall"] = evaluate_probe(narrative_probe, X_test, y_claims_test)

    if verbose:
        print(f"\n{'='*60}")
        print("OVERALL PROBE PERFORMANCE")
        print(f"{'='*60}")
        print(f"Reality probe accuracy:   {results['reality_overall'].accuracy:.1%}")
        print(f"Narrative probe accuracy: {results['narrative_overall'].accuracy:.1%}")

    # Per-category metrics
    results["reality_by_category"] = evaluate_by_category(
        reality_probe, X_test, y_tool_test, categories_test
    )
    results["narrative_by_category"] = evaluate_by_category(
        narrative_probe, X_test, y_claims_test, categories_test
    )

    if verbose:
        print_category_metrics(results["reality_by_category"], "Reality Probe")
        print_category_metrics(results["narrative_by_category"], "Narrative Probe")

    # Fake escalation analysis
    results["fake_analysis"] = analyze_fake_escalations(
        reality_probe, narrative_probe,
        X_test, y_tool_test, y_claims_test, categories_test
    )

    if verbose:
        print_fake_analysis(results["fake_analysis"])

    results["reality_probe"] = reality_probe
    results["narrative_probe"] = narrative_probe

    return results
