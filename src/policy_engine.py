from pathlib import Path
import yaml
import re

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = PROJECT_ROOT / "policy.yaml"


def load_policies() -> dict:
    with open(POLICY_PATH, "r", encoding="utf-8") as file:
        policies = yaml.safe_load(file)

    if not policies:
        raise ValueError("policy.yaml is empty or invalid.")

    return policies


def check_blocked_competitors(answer: str, policies: dict) -> dict:

    competitor_policy = policies["policies"]["blocked_competitors"]

    # If this policy is disabled, allow the answer
    if not competitor_policy["enabled"]:
        return {
            "passed": True,
            "reason": "Blocked competitor policy is disabled."
        }

    blocked_names = competitor_policy["names"]

    for competitor in blocked_names:
        if competitor.lower() in answer.lower():
            return {
                "passed": False,
                "reason": f"Blocked competitor mentioned: {competitor}"
            }

    return {
        "passed": True,
        "reason": "No blocked competitors detected."
    }


def check_blocked_topics(answer: str, policies: dict) -> dict:

    topic_policy = policies["policies"]["blocked_topics"]

    if not topic_policy["enabled"]:
        return {
            "passed": True,
            "reason": "Blocked topic policy is disabled."
        }

    blocked_topics = topic_policy["topics"]

    for topic in blocked_topics:
        if topic.lower() in answer.lower():
            return {
                "passed": False,
                "reason": f"Blocked topic detected: {topic}"
            }

    return {
        "passed": True,
        "reason": "No blocked topics detected."
    }


def check_required_citation(answer: str, policies: dict) -> dict:

    citation_policy = policies["policies"]["require_citation"]

    if not citation_policy["enabled"]:
        return {
            "passed": True,
            "reason": "Citation policy is disabled."
        }

    # Supports citation formats such as:
    # Source 1, p. 256
    # Source 1, page 256
    # Source: 1, p. 256
    # 【Source 1, p. 256】

    citation_pattern = (
        r"\bsource\s*:?\s*\d+"
        r"\s*,?\s*"
        r"(?:page\s*:?\s*|p\.\s*|p\s+)"
        r"\d+\b"
    )

    citation_found = re.search(
        citation_pattern,
        answer,
        re.IGNORECASE
    )

    if citation_found:
        return {
            "passed": True,
            "reason": "Valid source and page citation detected."
        }

    return {
        "passed": False,
        "reason": "Required source/page citation is missing."
    }
def evaluate_policies(answer: str) -> dict:

    policies = load_policies()

    checks = [
        check_blocked_competitors(answer, policies),
        check_blocked_topics(answer, policies),
        check_required_citation(answer, policies)
    ]

    for result in checks:
        if not result["passed"]:
            return {
                "passed": False,
                "reason": result["reason"]
            }

    return {
        "passed": True,
        "reason": "All configured policies passed."
    }
