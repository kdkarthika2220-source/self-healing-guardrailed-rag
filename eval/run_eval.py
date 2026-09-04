from pathlib import Path
import json
import sys
import time
import traceback
import random


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

EVAL_DATASET_PATH = (
    PROJECT_ROOT
    / "eval"
    / "eval_dataset.json"
)

EVAL_RESULTS_PATH = (
    PROJECT_ROOT
    / "eval"
    / "eval_results.json"
)


# ============================================================
# IMPORT RAG GRAPH + EXTERNAL JUDGE
# ============================================================

from self_healing_rag import graph
from judge import judge_rag_result


# ============================================================
# LOAD DATASET
# ============================================================

def load_eval_dataset() -> list[dict]:

    if not EVAL_DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation dataset not found: "
            f"{EVAL_DATASET_PATH}"
        )

    with open(
        EVAL_DATASET_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# LOAD EXISTING RESULTS
# ============================================================

def load_existing_results() -> list[dict]:

    if not EVAL_RESULTS_PATH.exists():
        return []

    with open(
        EVAL_RESULTS_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# MERGE RESULTS
# ============================================================

def merge_results(
    existing_results: list[dict],
    new_results: list[dict]
) -> list[dict]:

    merged = {
        result["id"]: result
        for result in existing_results
    }

    for result in new_results:
        merged[result["id"]] = result

    return sorted(
        merged.values(),
        key=lambda result: int(
            result["id"].replace(
                "q",
                ""
            )
        )
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    results: list[dict]
):

    with open(
        EVAL_RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# RETRY HELPER
# ============================================================

def run_with_retry(
    func,
    max_attempts: int = 4,
    base_delay: float = 1.0
):

    last_error = None

    for attempt in range(
        1,
        max_attempts + 1
    ):

        try:
            return func()

        except Exception as error:

            last_error = error

            error_text = str(
                error
            ).lower()

            transient_error = (
                "429" in error_text
                or "rate limit" in error_text
                or "connection error" in error_text
                or "timeout" in error_text
                or "temporarily unavailable" in error_text
            )

            if not transient_error:
                raise

            if attempt == max_attempts:
                raise

            delay = (
                base_delay
                * (2 ** (attempt - 1))
            )

            delay += random.uniform(
                0,
                0.5
            )

            print(
                f"\nTransient API error."
                f"\nRetry attempt : "
                f"{attempt}/{max_attempts - 1}"
                f"\nWaiting       : "
                f"{delay:.2f} sec"
            )

            time.sleep(
                delay
            )

    raise last_error


# ============================================================
# RUN SINGLE EVALUATION CASE
# ============================================================

def run_single_case(
    case: dict
) -> dict:

    question = case[
        "question"
    ]

    expected_behavior = case[
        "expected_behavior"
    ]

    initial_state = {
        "original_question": question,
        "retrieval_query": question,
        "sanitized_question": question,

        "context": "",
        "answer": "",

        "verdict": "",
        "reason": "",

        "retry_count": 0,

        # ====================================================
        # CRITIC INSTRUMENTATION
        # ====================================================

        "critic_not_grounded_count": 0,
        "critic_abstain_count": 0,

        "input_safe": True,
        "guardrail_reason": "",
    }

    start_time = time.perf_counter()

    try:

        # ====================================================
        # RUN RAG GRAPH
        # ====================================================

        final_state = run_with_retry(
            lambda: graph.invoke(
                initial_state
            )
        )

        rag_end_time = (
            time.perf_counter()
        )

        rag_latency = (
            rag_end_time
            - start_time
        )

        answer = final_state.get(
            "answer",
            ""
        )

        context = final_state.get(
            "context",
            ""
        )

        # ====================================================
        # EXTERNAL JUDGE
        # ====================================================

        judge_start_time = (
            time.perf_counter()
        )

        judge_result = run_with_retry(
            lambda: judge_rag_result(
                question=question,
                expected_behavior=expected_behavior,
                context=context,
                answer=answer
            )
        )

        judge_end_time = (
            time.perf_counter()
        )

        judge_latency = (
            judge_end_time
            - judge_start_time
        )

        total_eval_latency = (
            judge_end_time
            - start_time
        )

        # ====================================================
        # SUCCESS RESULT
        # ====================================================

        return {
            "id": case["id"],

            "category": case[
                "category"
            ],

            "question": question,

            "expected_behavior": (
                expected_behavior
            ),

            "expected_topic": (
                case.get(
                    "expected_topic"
                )
            ),

            "answer": answer,

            "verdict": final_state.get(
                "verdict",
                ""
            ),

            "reason": final_state.get(
                "reason",
                ""
            ),

            # =================================================
            # RETRY / CRITIC METRICS
            # =================================================

            "retry_count": final_state.get(
                "retry_count",
                0
            ),

            "critic_not_grounded_count": (
                final_state.get(
                    "critic_not_grounded_count",
                    0
                )
            ),

            "critic_abstain_count": (
                final_state.get(
                    "critic_abstain_count",
                    0
                )
            ),

            # =================================================
            # GUARDRAIL DATA
            # =================================================

            "input_safe": final_state.get(
                "input_safe",
                True
            ),

            "guardrail_reason": (
                final_state.get(
                    "guardrail_reason",
                    ""
                )
            ),

            "sanitized_question": (
                final_state.get(
                    "sanitized_question",
                    ""
                )
            ),

            # =================================================
            # EXTERNAL JUDGE RESULTS
            # =================================================

            "judge_behavior_correct": (
                judge_result.behavior_correct
            ),

            "judge_grounded": (
                judge_result.grounded
            ),

            "judge_relevant": (
                judge_result.relevant
            ),

            "judge_reason": (
                judge_result.reason
            ),

            # =================================================
            # LATENCY
            # =================================================

            "rag_latency_seconds": round(
                rag_latency,
                4
            ),

            "judge_latency_seconds": round(
                judge_latency,
                4
            ),

            "total_eval_latency_seconds": round(
                total_eval_latency,
                4
            ),

            "status": "SUCCESS",
        }

    except Exception as error:

        print(
            "\nEVALUATION ERROR:"
        )

        traceback.print_exc()

        total_eval_latency = (
            time.perf_counter()
            - start_time
        )

        # ====================================================
        # FAILED RESULT
        # ====================================================

        return {
            "id": case["id"],

            "category": case[
                "category"
            ],

            "question": question,

            "expected_behavior": (
                expected_behavior
            ),

            "expected_topic": (
                case.get(
                    "expected_topic"
                )
            ),

            "answer": "",

            "verdict": "",

            "reason": "",

            "retry_count": 0,

            # Keep the schema consistent even when
            # evaluation execution fails.

            "critic_not_grounded_count": 0,
            "critic_abstain_count": 0,

            "input_safe": True,

            "guardrail_reason": "",

            "sanitized_question": "",

            "judge_behavior_correct": False,

            "judge_grounded": False,

            "judge_relevant": False,

            "judge_reason": "",

            "rag_latency_seconds": 0,

            "judge_latency_seconds": 0,

            "total_eval_latency_seconds": round(
                total_eval_latency,
                4
            ),

            "status": "FAILED",

            "error": str(
                error
            ),
        }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "SELF-HEALING RAG EVALUATION"
    )

    print(
        "=" * 70
    )

    dataset = load_eval_dataset()

    critic_metric_case_ids = {
        "q02",
        "q05",
        "q06",
        "q09",
        "q10",
        "q11",
        "q12",
        "q13",
        "q19",
    }

    dataset = [
        case
        for case in dataset
        if case["id"] in critic_metric_case_ids
    ]

    print(
        f"\nEvaluation cases: "
        f"{len(dataset)}"
    )

    results = load_existing_results()

    for index, case in enumerate(
        dataset,
        start=1
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            f"CASE {index}/{len(dataset)}"
        )

        print(
            f"ID       : "
            f"{case['id']}"
        )

        print(
            f"Category : "
            f"{case['category']}"
        )

        print(
            f"Question : "
            f"{case['question']}"
        )

        print(
            "=" * 70
        )

        # ====================================================
        # RUN CASE
        # ====================================================

        result = run_single_case(
            case
        )

        # ====================================================
        # MERGE + SAVE
        # ====================================================

        results = merge_results(
            existing_results=results,
            new_results=[result]
        )

        save_results(
            results
        )

        # ====================================================
        # PRINT RESULT
        # ====================================================

        print(
            f"\nStatus            : "
            f"{result['status']}"
        )

        print(
            f"Internal verdict  : "
            f"{result['verdict']}"
        )

        print(
            f"Retries           : "
            f"{result['retry_count']}"
        )

        print(
            f"Critic NG catches : "
            f"{result['critic_not_grounded_count']}"
        )

        print(
            f"Critic abstains   : "
            f"{result['critic_abstain_count']}"
        )

        print(
            f"Judge behavior    : "
            f"{result['judge_behavior_correct']}"
        )

        print(
            f"Judge grounded    : "
            f"{result['judge_grounded']}"
        )

        print(
            f"Judge relevant    : "
            f"{result['judge_relevant']}"
        )

        print(
            f"RAG latency       : "
            f"{result['rag_latency_seconds']} sec"
        )

        print(
            f"Judge latency     : "
            f"{result['judge_latency_seconds']} sec"
        )

        print(
            f"Total eval latency: "
            f"{result['total_eval_latency_seconds']} sec"
        )

        print(
            "\nJudge reason:"
        )

        print(
            result[
                "judge_reason"
            ]
        )

        print(
            "\nAnswer:"
        )

        print(
            result[
                "answer"
            ]
        )

        # ====================================================
        # RATE LIMIT PACING
        # ====================================================

        if index < len(dataset):

            delay_between_cases = 5

            print(
                f"\nWaiting "
                f"{delay_between_cases} seconds "
                f"before next evaluation case..."
            )

            time.sleep(
                delay_between_cases
            )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "EVALUATION RUN COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nResults saved to:"
        f"\n{EVAL_RESULTS_PATH}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()