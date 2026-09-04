from pathlib import Path
import json
import statistics


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    PROJECT_ROOT
    / "eval"
    / "eval_results.json"
)


# ============================================================
# LOAD RESULTS
# ============================================================

def load_results() -> list[dict]:

    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation results not found: {RESULTS_PATH}"
        )

    with open(
        RESULTS_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


# ============================================================
# PERCENTAGE
# ============================================================

def percentage(
    numerator: int,
    denominator: int
) -> float:

    if denominator == 0:
        return 0.0

    return round(
        numerator / denominator * 100,
        2
    )


# ============================================================
# P95
# ============================================================

def percentile_95(
    values: list[float]
) -> float:

    if not values:
        return 0.0

    if len(values) == 1:
        return round(
            values[0],
            4
        )

    return round(
        statistics.quantiles(
            values,
            n=100,
            method="inclusive"
        )[94],
        4
    )


# ============================================================
# CALCULATE METRICS
# ============================================================

def calculate_metrics(
    results: list[dict]
) -> dict:

    successful = [
        result
        for result in results
        if result.get("status") == "SUCCESS"
    ]

    total = len(results)

    successful_count = len(
        successful
    )

    # ========================================================
    # EXTERNAL JUDGE METRICS
    # ========================================================

    behavior_correct_count = sum(
        bool(
            result.get(
                "judge_behavior_correct"
            )
        )
        for result in successful
    )

    grounded_count = sum(
        bool(
            result.get(
                "judge_grounded"
            )
        )
        for result in successful
    )

    relevant_count = sum(
        bool(
            result.get(
                "judge_relevant"
            )
        )
        for result in successful
    )

    # ========================================================
    # ANSWER-BEARING GROUNDEDNESS
    # ========================================================

    answer_behaviors = {
        "answer",
        "partial_answer",
        "answer_after_redaction",
    }

    answer_cases = [
        result
        for result in successful
        if result.get(
            "expected_behavior"
        ) in answer_behaviors
    ]

    grounded_answer_cases = sum(
        bool(
            result.get(
                "judge_grounded"
            )
        )
        for result in answer_cases
    )

    # ========================================================
    # ABSTENTION ACCURACY
    # ========================================================

    abstain_cases = [
        result
        for result in successful
        if result.get(
            "expected_behavior"
        ) == "abstain"
    ]

    correct_abstentions = sum(
        result.get(
            "verdict"
        ) == "ABSTAINED"
        for result in abstain_cases
    )

    # ========================================================
    # PROMPT INJECTION BLOCK RATE
    # ========================================================

    injection_cases = [
        result
        for result in successful
        if result.get(
            "category"
        ) == "prompt_injection"
    ]

    blocked_injections = sum(
        result.get(
            "input_safe"
        ) is False
        for result in injection_cases
    )

    # ========================================================
    # PII REDACTION SUCCESS
    # ========================================================

    pii_cases = [
        result
        for result in successful
        if result.get(
            "category"
        ) == "pii"
    ]

    pii_success = sum(
        bool(
            result.get(
                "judge_behavior_correct"
            )
        )
        for result in pii_cases
    )

    # ========================================================
    # RETRY METRICS
    # ========================================================

    retry_counts = [
        result.get(
            "retry_count",
            0
        )
        for result in successful
    ]

    cases_with_retry = sum(
        retry_count > 0
        for retry_count in retry_counts
    )

    average_retries = (
        statistics.mean(
            retry_counts
        )
        if retry_counts
        else 0.0
    )

    # ========================================================
    # OVERALL RAG LATENCY
    # ========================================================

    rag_latencies = [
        result.get(
            "rag_latency_seconds",
            0
        )
        for result in successful
        if result.get(
            "rag_latency_seconds",
            0
        ) > 0
    ]

    average_latency = (
        statistics.mean(
            rag_latencies
        )
        if rag_latencies
        else 0.0
    )

    p50_latency = (
        statistics.median(
            rag_latencies
        )
        if rag_latencies
        else 0.0
    )

    p95_latency = percentile_95(
        rag_latencies
    )

    # ========================================================
    # NON-BLOCKED RAG LATENCY
    # ========================================================

    non_blocked_latencies = [
        result.get(
            "rag_latency_seconds",
            0
        )
        for result in successful
        if (
            result.get(
                "input_safe"
            ) is not False
            and result.get(
                "rag_latency_seconds",
                0
            ) > 0
        )
    ]

    non_blocked_mean = (
        statistics.mean(
            non_blocked_latencies
        )
        if non_blocked_latencies
        else 0.0
    )

    non_blocked_p50 = (
        statistics.median(
            non_blocked_latencies
        )
        if non_blocked_latencies
        else 0.0
    )

    non_blocked_p95 = percentile_95(
        non_blocked_latencies
    )

    # ========================================================
    # INPUT GUARDRAIL BLOCK LATENCY
    # ========================================================

    blocked_latencies = [
        result.get(
            "rag_latency_seconds",
            0
        )
        for result in successful
        if (
            result.get(
                "input_safe"
            ) is False
            and result.get(
                "rag_latency_seconds",
                0
            ) > 0
        )
    ]

    blocked_mean = (
        statistics.mean(
            blocked_latencies
        )
        if blocked_latencies
        else 0.0
    )

    # ========================================================
    # FAILURE CASES
    # ========================================================

    judge_failures = [
        {
            "id": result.get(
                "id"
            ),

            "behavior_correct": result.get(
                "judge_behavior_correct"
            ),

            "grounded": result.get(
                "judge_grounded"
            ),

            "relevant": result.get(
                "judge_relevant"
            ),

            "reason": result.get(
                "judge_reason",
                ""
            ),
        }

        for result in successful

        if (
            not result.get(
                "judge_behavior_correct"
            )
            or not result.get(
                "judge_grounded"
            )
            or not result.get(
                "judge_relevant"
            )
        )
    ]

    # ========================================================
    # RETURN METRICS
    # ========================================================

    return {
        "total_cases": total,

        "successful_cases": successful_count,

        "execution_success_rate": percentage(
            successful_count,
            total
        ),

        "behavior_accuracy": percentage(
            behavior_correct_count,
            successful_count
        ),

        "grounded_rate": percentage(
            grounded_count,
            successful_count
        ),

        "relevance_rate": percentage(
            relevant_count,
            successful_count
        ),

        "answer_grounded_rate": percentage(
            grounded_answer_cases,
            len(answer_cases)
        ),

        "answer_bearing_cases": len(
            answer_cases
        ),

        "abstention_accuracy": percentage(
            correct_abstentions,
            len(abstain_cases)
        ),

        "prompt_injection_block_rate": percentage(
            blocked_injections,
            len(injection_cases)
        ),

        "pii_redaction_success_rate": percentage(
            pii_success,
            len(pii_cases)
        ),

        "retry_rate": percentage(
            cases_with_retry,
            successful_count
        ),

        "average_retries": round(
            average_retries,
            2
        ),

        "rag_latency_mean_seconds": round(
            average_latency,
            4
        ),

        "rag_latency_p50_seconds": round(
            p50_latency,
            4
        ),

        "rag_latency_p95_seconds": (
            p95_latency
        ),

        "non_blocked_latency_mean_seconds": round(
            non_blocked_mean,
            4
        ),

        "non_blocked_latency_p50_seconds": round(
            non_blocked_p50,
            4
        ),

        "non_blocked_latency_p95_seconds": (
            non_blocked_p95
        ),

        "guardrail_block_latency_mean_seconds": round(
            blocked_mean,
            4
        ),

        "judge_failure_cases": (
            judge_failures
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main():

    results = load_results()

    metrics = calculate_metrics(
        results
    )

    print(
        "\n"
        + "=" * 60
    )

    print(
        "RAG EVALUATION METRICS"
    )

    print(
        "=" * 60
    )

    # ========================================================
    # EXECUTION
    # ========================================================

    print(
        f"\nTotal cases              : "
        f"{metrics['total_cases']}"
    )

    print(
        f"Successful executions    : "
        f"{metrics['successful_cases']}"
    )

    print(
        f"Execution success rate   : "
        f"{metrics['execution_success_rate']}%"
    )

    # ========================================================
    # QUALITY
    # ========================================================

    print(
        f"\nBehavior accuracy        : "
        f"{metrics['behavior_accuracy']}%"
    )

    print(
        f"Grounded rate            : "
        f"{metrics['grounded_rate']}%"
    )

    print(
        f"Answer grounded rate     : "
        f"{metrics['answer_grounded_rate']}%"
    )

    print(
        f"Answer-bearing cases     : "
        f"{metrics['answer_bearing_cases']}"
    )

    print(
        f"Relevance rate           : "
        f"{metrics['relevance_rate']}%"
    )

    # ========================================================
    # SAFETY
    # ========================================================

    print(
        f"\nAbstention accuracy      : "
        f"{metrics['abstention_accuracy']}%"
    )

    print(
        f"Injection block rate     : "
        f"{metrics['prompt_injection_block_rate']}%"
    )

    print(
        f"PII redaction success    : "
        f"{metrics['pii_redaction_success_rate']}%"
    )

    # ========================================================
    # RETRIES
    # ========================================================

    print(
        f"\nRetry rate               : "
        f"{metrics['retry_rate']}%"
    )

    print(
        f"Average retries          : "
        f"{metrics['average_retries']}"
    )

    # ========================================================
    # OVERALL LATENCY
    # ========================================================

    print(
        "\nOverall RAG latency:"
    )

    print(
        f"Mean                     : "
        f"{metrics['rag_latency_mean_seconds']} sec"
    )

    print(
        f"P50                      : "
        f"{metrics['rag_latency_p50_seconds']} sec"
    )

    print(
        f"P95                      : "
        f"{metrics['rag_latency_p95_seconds']} sec"
    )

    # ========================================================
    # NON-BLOCKED LATENCY
    # ========================================================

    print(
        "\nNon-blocked RAG latency:"
    )

    print(
        f"Mean                     : "
        f"{metrics['non_blocked_latency_mean_seconds']} sec"
    )

    print(
        f"P50                      : "
        f"{metrics['non_blocked_latency_p50_seconds']} sec"
    )

    print(
        f"P95                      : "
        f"{metrics['non_blocked_latency_p95_seconds']} sec"
    )

    # ========================================================
    # GUARDRAIL BLOCK LATENCY
    # ========================================================

    print(
        "\nInput guardrail latency:"
    )

    print(
        f"Mean                     : "
        f"{metrics['guardrail_block_latency_mean_seconds']} sec"
    )

    # ========================================================
    # FAILURE CASES
    # ========================================================

    print(
        "\nJudge failure cases:"
    )

    if not metrics[
        "judge_failure_cases"
    ]:

        print(
            "None"
        )

    else:

        for failure in metrics[
            "judge_failure_cases"
        ]:

            print(
                f"\n{failure['id']}"
            )

            print(
                f"Behavior : "
                f"{failure['behavior_correct']}"
            )

            print(
                f"Grounded : "
                f"{failure['grounded']}"
            )

            print(
                f"Relevant : "
                f"{failure['relevant']}"
            )

            print(
                f"Reason   : "
                f"{failure['reason']}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()