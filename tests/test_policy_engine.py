from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from policy_engine import evaluate_policies


def test_valid_answer_passes_policy():

    answer = (
        "A Digital Health Platform provides digital "
        "infrastructure for health services "
        "【Source 1, p. 10】"
    )

    result = evaluate_policies(answer)

    assert result["passed"] is True


def test_answer_without_citation_fails():

    answer = (
        "A Digital Health Platform provides "
        "digital infrastructure."
    )

    result = evaluate_policies(answer)

    assert result["passed"] is False


def test_blocked_competitor_fails():

    answer = (
        "CompetitorX provides this service "
        "【Source 1, p. 10】"
    )

    result = evaluate_policies(answer)

    assert result["passed"] is False


def test_medical_treatment_advice_topic_fails():

    answer = (
        "This response contains medical treatment advice "
        "【Source 1, p. 10】"
    )

    result = evaluate_policies(answer)

    assert result["passed"] is False