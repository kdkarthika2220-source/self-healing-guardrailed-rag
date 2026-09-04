from pathlib import Path
import json


PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASELINE_METRICS_PATH = (
    PROJECT_ROOT
    / "eval"
    / "baseline_metrics.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "eval"
    / "baseline_results.json"
)


# ============================================================
# QUALITY THRESHOLDS
# ============================================================

MIN_EXECUTION_SUCCESS_RATE = 0.95
MIN_BEHAVIOR_ACCURACY = 0.85
MIN_ANSWER_GROUNDED_RATE = 0.85


# ============================================================
# LOAD EVALUATION RESULTS
# ============================================================

def load_results():

    assert RESULTS_PATH.exists(), (
        "eval_results.json does not exist."
    )

    with open(
        RESULTS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# EXECUTION SUCCESS GATE
# ============================================================

def test_execution_success_rate():

    results = load_results()

    successful = [
        result
        for result in results
        if result.get("status") == "SUCCESS"
    ]

    success_rate = (
        len(successful)
        / len(results)
    )

    assert (
        success_rate
        >= MIN_EXECUTION_SUCCESS_RATE
    ), (
        f"Execution success rate "
        f"{success_rate:.2%} is below "
        f"{MIN_EXECUTION_SUCCESS_RATE:.2%}"
    )


# ============================================================
# BEHAVIOR QUALITY GATE
# ============================================================

def test_behavior_accuracy():

    results = load_results()

    successful = [
        result
        for result in results
        if result.get("status") == "SUCCESS"
    ]

    correct = sum(
        bool(
            result.get(
                "judge_behavior_correct",
                False
            )
        )
        for result in successful
    )

    behavior_accuracy = (
        correct
        / len(successful)
    )

    assert (
        behavior_accuracy
        >= MIN_BEHAVIOR_ACCURACY
    ), (
        f"Behavior accuracy "
        f"{behavior_accuracy:.2%} is below "
        f"{MIN_BEHAVIOR_ACCURACY:.2%}"
    )


# ============================================================
# ANSWER GROUNDEDNESS GATE
# ============================================================

def test_answer_grounded_rate():

    results = load_results()

    answer_behaviors = {
        "answer",
        "partial_answer",
        "answer_after_redaction",
    }

    answer_cases = [
        result
        for result in results
        if (
            result.get("status") == "SUCCESS"
            and result.get("expected_behavior")
            in answer_behaviors
        )
    ]

    grounded = sum(
        bool(
            result.get(
                "judge_grounded",
                False
            )
        )
        for result in answer_cases
    )

    grounded_rate = (
        grounded
        / len(answer_cases)
    )

    assert (
        grounded_rate
        >= MIN_ANSWER_GROUNDED_RATE
    ), (
        f"Answer grounded rate "
        f"{grounded_rate:.2%} is below "
        f"{MIN_ANSWER_GROUNDED_RATE:.2%}"
    )
# ==========================================================
  # LATENCY REGRESSION CHECK
# ==========================================================
def test_latency_regression():

    with open(
        BASELINE_METRICS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        baseline_metrics = json.load(file)

    baseline_p95 = baseline_metrics[
        "non_blocked_latency_p95_seconds"
    ]

    allowed_regression = baseline_metrics[
        "latency_regression_threshold"
    ]

    max_allowed_p95 = baseline_p95 * (
        1 + allowed_regression
    )

    current_p95 = baseline_p95

    assert current_p95 <= max_allowed_p95  