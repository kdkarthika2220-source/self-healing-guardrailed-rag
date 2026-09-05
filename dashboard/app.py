from pathlib import Path
import json
import statistics

import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Self-Healing RAG Dashboard",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = (
    PROJECT_ROOT
    / "eval"
    / "baseline_results.json"
)


# ============================================================
# LOAD EVALUATION RESULTS
# ============================================================

@st.cache_data
def load_results():
    """
    Load the approved evaluation baseline.

    st.cache_data prevents Streamlit from reading the JSON file
    again on every UI rerender.
    """

    with open(
        RESULTS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


results = load_results()


# ============================================================
# FILTER SUCCESSFUL EXECUTIONS
# ============================================================

successful_results = [
    result
    for result in results
    if result["status"] == "SUCCESS"
]

total_cases = len(results)
successful_cases = len(successful_results)


# ============================================================
# QUALITY METRICS
# ============================================================

execution_success_rate = (
    successful_cases
    / total_cases
    * 100
)


behavior_accuracy = (
    sum(
        result["judge_behavior_correct"]
        for result in successful_results
    )
    / successful_cases
    * 100
)


grounded_rate = (
    sum(
        result["judge_grounded"]
        for result in successful_results
    )
    / successful_cases
    * 100
)


# ------------------------------------------------------------
# Answer-bearing grounding
#
# Overall grounding also includes abstention / blocking cases.
# This metric evaluates only cases where the RAG system
# was expected to produce an actual answer.
# ------------------------------------------------------------

answer_behaviors = {
    "answer",
    "partial_answer",
    "answer_after_redaction",
}

answer_results = [
    result
    for result in successful_results
    if result["expected_behavior"] in answer_behaviors
]

answer_grounded_rate = (
    sum(
        result["judge_grounded"]
        for result in answer_results
    )
    / len(answer_results)
    * 100
)


# ============================================================
# SELF-HEALING METRICS
# ============================================================

retry_rate = (
    sum(
        result["retry_count"] > 0
        for result in successful_results
    )
    / successful_cases
    * 100
)


# ------------------------------------------------------------
# Abstention accuracy
# ------------------------------------------------------------

abstain_results = [
    result
    for result in successful_results
    if result["expected_behavior"] == "abstain"
]

abstention_accuracy = (
    sum(
        result["verdict"] == "ABSTAINED"
        for result in abstain_results
    )
    / len(abstain_results)
    * 100
)


# ============================================================
# LATENCY METRICS
# ============================================================

# Exclude requests stopped immediately by the input guardrail.
# Otherwise those millisecond-level cases would artificially
# reduce our RAG latency statistics.

non_blocked_results = [
    result
    for result in successful_results
    if result["input_safe"] is not False
]

latencies = [
    result["rag_latency_seconds"]
    for result in non_blocked_results
]


# ------------------------------------------------------------
# P50 latency
# ------------------------------------------------------------

p50_latency = statistics.median(
    latencies
)


# ------------------------------------------------------------
# P95 latency
#
# This is a simple percentile calculation for dashboard display.
# Our official evaluation metrics should remain the source of
# truth when comparing benchmark runs.
# ------------------------------------------------------------

sorted_latencies = sorted(
    latencies
)

p95_index = int(
    0.95 * (len(sorted_latencies) - 1)
)

p95_latency = sorted_latencies[
    p95_index
]


# ============================================================
# RETRY LATENCY ANALYSIS
# ============================================================

retry_latencies = [
    result["rag_latency_seconds"]
    for result in successful_results
    if result["retry_count"] > 0
]

no_retry_latencies = [
    result["rag_latency_seconds"]
    for result in successful_results
    if (
        result["retry_count"] == 0
        and result["input_safe"] is not False
    )
]


average_retry_latency = (
    sum(retry_latencies)
    / len(retry_latencies)
)

average_no_retry_latency = (
    sum(no_retry_latencies)
    / len(no_retry_latencies)
)


# ============================================================
# GUARDRAIL METRICS
# ============================================================

# ------------------------------------------------------------
# Prompt injection block rate
# ------------------------------------------------------------

injection_results = [
    result
    for result in successful_results
    if result["expected_behavior"] == "block"
]

injection_block_rate = (
    sum(
        result["input_safe"] is False
        for result in injection_results
    )
    / len(injection_results)
    * 100
)


# ------------------------------------------------------------
# PII redaction success
#
# Important:
# This measures expected-behavior success for the current
# redaction evaluation cases. It is NOT a complete benchmark
# of the underlying PII detector's accuracy.
# ------------------------------------------------------------

pii_results = [
    result
    for result in successful_results
    if result["expected_behavior"]
    == "answer_after_redaction"
]

pii_redaction_success_rate = (
    sum(
        result["judge_behavior_correct"]
        for result in pii_results
    )
    / len(pii_results)
    * 100
)


# ============================================================
# DASHBOARD HEADER
# ============================================================

st.title(
    "Self-Healing RAG Evaluation Dashboard"
)

st.caption(
    "Evaluation metrics for retrieval quality, "
    "self-healing behavior, guardrails, and latency."
)

st.write(
    f"Evaluation cases loaded: {total_cases}"
)


# ============================================================
# 1. EVALUATION OVERVIEW
# ============================================================

st.header(
    "Evaluation Overview"
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Execution Success",
    f"{execution_success_rate:.1f}%",
)

col2.metric(
    "Behavior Accuracy",
    f"{behavior_accuracy:.1f}%",
)

col3.metric(
    "Overall Grounded Rate",
    f"{grounded_rate:.1f}%",
)


st.metric(
    "Answer Grounded Rate",
    f"{answer_grounded_rate:.2f}%",
    help=(
        "Grounding rate calculated only for evaluation "
        "cases where the system was expected to answer."
    ),
)


# ============================================================
# 2. SELF-HEALING PERFORMANCE
# ============================================================

st.header(
    "Self-Healing Performance"
)

col4, col5 = st.columns(2)

col4.metric(
    "Retry Rate",
    f"{retry_rate:.1f}%",
)

col5.metric(
    "Abstention Accuracy",
    f"{abstention_accuracy:.1f}%",
)


st.subheader(
    "Retry Impact on Latency"
)

col6, col7 = st.columns(2)

col6.metric(
    "Avg Latency — Retry",
    f"{average_retry_latency:.2f} sec",
)

col7.metric(
    "Avg Latency — No Retry",
    f"{average_no_retry_latency:.2f} sec",
)


# ============================================================
# 3. LATENCY ANALYSIS
# ============================================================

st.header(
    "Latency Analysis"
)

col8, col9 = st.columns(2)

col8.metric(
    "Latency P50",
    f"{p50_latency:.2f} sec",
)

col9.metric(
    "Latency P95",
    f"{p95_latency:.2f} sec",
)


# ------------------------------------------------------------
# Per-evaluation-case latency visualization
# ------------------------------------------------------------

st.subheader(
    "RAG Latency by Evaluation Case"
)

latency_chart_data = {
    result["id"]: result["rag_latency_seconds"]
    for result in successful_results
}

st.bar_chart(
    latency_chart_data,
    x_label="Evaluation Case",
    y_label="Latency (seconds)",
)


# ============================================================
# 4. GUARDRAIL PERFORMANCE
# ============================================================

st.header(
    "Guardrail Performance"
)

col10, col11 = st.columns(2)

col10.metric(
    "Prompt Injection Block Rate",
    f"{injection_block_rate:.1f}%",
    help=(
        "Percentage of the current prompt-injection "
        "evaluation cases blocked by the input guardrail."
    ),
)

col11.metric(
    "PII Redaction Success Rate",
    f"{pii_redaction_success_rate:.1f}%",
    help=(
        "Expected-behavior success rate for the current "
        "PII redaction evaluation cases."
    ),
)


# ============================================================
# 5. FAILURE ANALYSIS
# ============================================================

st.header(
    "Failure Analysis"
)


# ------------------------------------------------------------
# Identify evaluation cases that failed one or more
# quality dimensions.
# ------------------------------------------------------------

failed_quality_cases = [
    result
    for result in successful_results
    if (
        not result["judge_behavior_correct"]
        or not result["judge_grounded"]
        or not result["judge_relevant"]
    )
]


st.subheader(
    "Evaluation Failures"
)

if failed_quality_cases:

    failure_table = []

    for result in failed_quality_cases:

        failure_table.append(
            {
                "Case ID":
                    result["id"],

                "Expected Behavior":
                    result["expected_behavior"],

                # Convert booleans to strings so Streamlit
                # displays True / False instead of checkboxes.
                "Behavior Correct":
                    str(
                        result[
                            "judge_behavior_correct"
                        ]
                    ),

                "Grounded":
                    str(
                        result[
                            "judge_grounded"
                        ]
                    ),

                "Relevant":
                    str(
                        result[
                            "judge_relevant"
                        ]
                    ),

                "Retries":
                    result["retry_count"],
            }
        )

    st.dataframe(
        failure_table,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.success(
        "No evaluation quality failures found."
    )


# ============================================================
# FAILURE CATEGORY VISUALIZATION
# ============================================================

st.subheader(
    "Failure Categories"
)

failure_categories = {
    "Behavior Failure": sum(
        not result["judge_behavior_correct"]
        for result in successful_results
    ),

    "Grounding Failure": sum(
        not result["judge_grounded"]
        for result in successful_results
    ),

    "Relevance Failure": sum(
        not result["judge_relevant"]
        for result in successful_results
    ),
}

st.bar_chart(
    failure_categories,
    x_label="Failure Category",
    y_label="Number of Cases",
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Metrics are generated from the approved evaluation "
    "baseline stored in eval/baseline_results.json."
)