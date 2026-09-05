from typing import TypedDict
import os
import time

from guardrails import (
    detect_prompt_injection,
    detect_and_redact_pii,
    validate_rag_output,
    check_output_safety,
)
from retrieval import hybrid_search
from policy_engine import evaluate_policies
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from groq import Groq

class CriticResult(BaseModel):

    verdict: str = Field(
        description="Must be either GROUNDED or NOT_GROUNDED"
    )

    reason: str = Field(
        description="Short explanation for the verdict"
    )

GROQ_MODEL = "openai/gpt-oss-120b"

groq_client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)



# ============================================================
# LANGGRAPH STATE
# ============================================================

class RAGState(TypedDict, total=False):

    question: str
    sanitized_question: str

    retrieval_query: str
    context: str

    answer: str

    verdict: str
    critic_reason: str

    input_safe: bool
    guardrail_reason: str

    output_safe: bool
    output_guardrail_reason: str

    policy_passed: bool
    policy_reason: str

    retry_count: int
    status: str
# ===========================================================
# INPUT GUARDRAIL NODE
# ============================================================
def input_guardrail_node(state: RAGState):

    print("\n" + "=" * 60)
    print("INPUT GUARDRAIL NODE")
    print("=" * 60)

    user_input = state["sanitized_question"]

    # --------------------------------------------------------
    # 1. PROMPT INJECTION CHECK
    # --------------------------------------------------------

    injection_result = detect_prompt_injection(user_input)

    print(f"Prompt injection safe : {injection_result['is_safe']}")
    print(f"Reason                : {injection_result['reason']}")

    # Malicious input → block immediately
    if not injection_result["is_safe"]:
        return {
            "input_safe": False,
            "guardrail_reason": injection_result["reason"],
            "sanitized_question": ""
        }

    # --------------------------------------------------------
    # 2. PII DETECTION + REDACTION
    # --------------------------------------------------------

    pii_result = detect_and_redact_pii(user_input)

    if pii_result["pii_detected"]:

        print(f"PII detected           : True")
        print(f"PII entities           : {pii_result['entities']}")
        print(f"Sanitized input        : {pii_result['redacted_text']}")

    else:
        print("PII detected           : False")

    sanitized_question = pii_result["redacted_text"]

    return {
        "input_safe": True,
        "guardrail_reason": "Input passed safety checks.",
        "sanitized_question": sanitized_question,
        "retrieval_query": sanitized_question
    }
# ============================================================
# ROUTE AFTER INPUT GUARDRAIL
# ============================================================
def route_after_input_guardrail(state: RAGState):

    print("\n" + "=" * 60)
    print("INPUT GUARDRAIL ROUTING")
    print("=" * 60)

    if state["input_safe"]:
        print("Decision : ALLOW → RETRIEVE")
        return "allow"

    print("Decision : BLOCK")
    return "block"
# ============================================================
# BLOCKED INPUT  NODE
# ============================================================

def blocked_input_node(state: RAGState):

    print("\n" + "=" * 60)
    print("BLOCKED INPUT NODE")
    print("=" * 60)

    response = (
        "I can't process this request because it violates the input safety policy."
    )

    print(f"Final response:\n{response}")

    return {
        "answer": response
    }
# ============================================================
# RETRIEVE NODE
# ============================================================

def retrieve_node(state: RAGState):

    print("\n" + "=" * 60)
    print("HYBRID RETRIEVE NODE")
    print("=" * 60)

    query = state["retrieval_query"]

    print(f"Query: {query}")

    start_time = time.perf_counter()

    # --------------------------------------------------------
    # Hybrid retrieval
    # --------------------------------------------------------

    output = hybrid_search(
        query=query
    )

    results = output["results"]

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context_parts = []

    for i, result in enumerate(
        results,
        start=1
    ):

        metadata = result["metadata"]

        page = metadata.get(
            "page",
            "Unknown"
        )

        chunk_index = metadata.get(
            "chunk_index",
            "Unknown"
        )

        retrievers = result.get(
            "retrievers",
            []
        )

        rrf_score = result.get(
            "rrf_score",
            0
        )

        document = result[
            "document"
        ]

        context_parts.append(
            f"SOURCE {i}\n"
            f"Page: {page}\n"
            f"Chunk: {chunk_index}\n"
            f"Retrieved by: {retrievers}\n"
            f"RRF score: {rrf_score:.6f}\n"
            f"Content:\n{document}"
        )

    context = "\n\n".join(
        context_parts
    )

    total_time = (
        time.perf_counter()
        - start_time
    )

    # --------------------------------------------------------
    # Timing information
    # --------------------------------------------------------

    timing = output["timing"]

    print(
        f"\nEmbedding time : "
        f"{timing['embedding_time']:.4f} sec"
    )

    print(
        f"Vector search  : "
        f"{timing['vector_retrieval_time']:.4f} sec"
    )

    print(
        f"BM25 search    : "
        f"{timing['bm25_time']:.4f} sec"
    )

    print(
        f"RRF fusion     : "
        f"{timing['fusion_time']:.6f} sec"
    )

    print(
        f"Total retrieve : "
        f"{total_time:.4f} sec"
    )

    print("\nRetrieved context:")
    print("-" * 60)

    print(
        context[:2000]
    )

    print("-" * 60)

    return {
        "context": context
    }
# ============================================================
# GENERATE NODE
# ============================================================

def generate_node(state: RAGState):

    print("\n" + "=" * 60)
    print("GENERATE NODE")
    print("=" * 60)

    question = state["sanitized_question"]
    context = state["context"]

    start_time = time.perf_counter()

    system_prompt = """

You are a question-answering assistant.

Answer the user's question ONLY using the provided context.

Rules:

1. Do not use outside knowledge.

2. Do not invent, assume, or infer information that is not
   supported by the provided context.

3. A user question may contain multiple parts.

4. If the context supports only some parts of the question,
   answer those supported parts normally.

5. For each unsupported part, clearly state that the provided
   document does not contain enough information to answer
   that part.

6. Do NOT refuse the entire question when at least one part
   can be answered from the context.

7. If NONE of the question can be answered from the context,
   you MUST return exactly this sentence and nothing else:

   "I don't have enough information in the provided document."

   Do not paraphrase this sentence.

8. Keep the answer concise and factual.

9. Mention the relevant page number when possible.


CITATION RULES:

- Every factual answer must include citations using ONLY this format:
  【Source N, p. P】

- N must be the source number shown in the retrieved context.
- P must be the page number shown in the retrieved context.

Example:
【Source 2, p. 45】

- Do NOT use citation formats such as:
  【3†L7-L12】
  [3]
  (source 3)
  or any other citation style.

- Never invent a source number or page number.
- Only cite source/page values explicitly present in the retrieved context.

"""

    user_prompt = f"""
QUESTION:
{question}

CONTEXT:
{context}
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0,
        reasoning_effort="low"
    )

    answer = response.choices[0].message.content.strip()

    

    generation_time = time.perf_counter() - start_time

    print("\nGenerated answer:")
    print(answer)

    print(f"\nGeneration time: {generation_time:.4f} sec")

    return {
        "answer": answer
    }
# ============================================================
# CRITIC NODE
# ============================================================
def critic_node(state: RAGState):

    print("\n" + "=" * 60)
    print("CRITIC NODE")
    print("=" * 60)

    question = state["sanitized_question"]
    context = state["context"]
    answer = state["answer"]
# ============================================================
# CHECK FOR GENERATOR ABSTENTION
# ============================================================

    abstention_answer = (
    "I don't have enough information "
    "in the provided document."
)

    if (
        answer.strip().lower()
        == abstention_answer.lower()
    ):

        print("\nVerdict:")
        print("ABSTAINED")

        print("\nReason:")
        print(
            "The generated answer is an abstention because the "
            "context does not contain enough information."
        )

        return {
            "verdict": "ABSTAINED",

            "reason": (
                "The context does not contain enough information "
                "to answer the question."
            ),

            "critic_abstain_count": (
                state.get(
                    "critic_abstain_count",
                    0
                )
                + 1
            )
        }
    start_time = time.perf_counter()

    system_prompt = """
You are a strict RAG evaluator.

Determine whether the ANSWER is fully supported by the
provided CONTEXT.

Rules:

1. Use ONLY the provided context.
2. Do not use outside knowledge.
3. If the answer contains information not supported by
   the context, classify it as NOT_GROUNDED.
4. If the context does not contain enough information,
   classify it as NOT_GROUNDED.
5. If the answer is directly supported by the context,
   classify it as GROUNDED.
6. Be strict.

The verdict must be exactly one of:

GROUNDED
NOT_GROUNDED
"""

    user_prompt = f"""
QUESTION:
{question}

CONTEXT:
{context}

ANSWER:
{answer}
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0,
        reasoning_effort="low",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "critic_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "verdict": {
                            "type": "string",
                            "enum": [
                                "GROUNDED",
                                "NOT_GROUNDED"
                            ]
                        },
                        "reason": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "verdict",
                        "reason"
                    ],
                    "additionalProperties": False
                }
            }
        }
    )

    raw_result = response.choices[0].message.content

    critic_result = CriticResult.model_validate_json(
        raw_result
    )

    elapsed = time.perf_counter() - start_time

    print("\nVerdict:")
    print(critic_result.verdict)

    print("\nReason:")
    print(critic_result.reason)

    print(f"\nCritic time: {elapsed:.4f} sec")

    result = {
    "verdict": critic_result.verdict,
    "reason": critic_result.reason
}

    if critic_result.verdict == "NOT_GROUNDED":


        result["critic_not_grounded_count"] = (
            state.get(
                "critic_not_grounded_count",
                0
            )
            + 1
        )

    return result
def output_guardrail_node(state: RAGState):

    print("\n" + "=" * 60)
    print("OUTPUT GUARDRAIL NODE")
    print("=" * 60)

    answer = state["answer"]

    # ========================================================
    # 1. SCHEMA VALIDATION
    # ========================================================

    schema_result = validate_rag_output(answer)

    print(f"Schema valid : {schema_result['valid']}")
    print(f"Reason       : {schema_result['reason']}")

    if not schema_result["valid"]:
        return {
            "answer": "I couldn't return a valid response safely.",
            "output_safe": False,
            "output_guardrail_reason": schema_result["reason"],
            "status": "OUTPUT_BLOCKED"
        }

    # ========================================================
    # 2. OUTPUT SAFETY CHECK
    # ========================================================

    safety_result = check_output_safety(
        schema_result["answer"]
    )

    print(f"Output safe  : {safety_result['safe']}")
    print(f"Reason       : {safety_result['reason']}")

    if not safety_result["safe"]:
        return {
            "answer": "I couldn't return this response because it did not pass the output safety check.",
            "output_safe": False,
            "output_guardrail_reason": safety_result["reason"],
            "status": "OUTPUT_BLOCKED"
        }

    return {
        "answer": schema_result["answer"],
        "output_safe": True,
        "output_guardrail_reason": "Output passed all guardrail checks."
    }
# ============================================================
# POLICY ENGINE NODE
# ============================================================

def policy_node(state: RAGState):

    print("\n" + "=" * 60)
    print("POLICY ENGINE NODE")
    print("=" * 60)

    answer = state["answer"]

    policy_result = evaluate_policies(answer)

    print(f"Policy passed : {policy_result['passed']}")
    print(f"Reason        : {policy_result['reason']}")

    if not policy_result["passed"]:
        return {
            "answer": "I couldn't return this response because it violates the configured policy.",
            "policy_passed": False,
            "policy_reason": policy_result["reason"],
            "status": "POLICY_BLOCKED"
        }

    return {
        "answer": answer,
        "policy_passed": True,
        "policy_reason": policy_result["reason"],
        "status": "SUCCESS"
    }

# ============================================================
# ROUTING NODE
# ============================================================
def route_after_critic(state: RAGState):

    verdict = state["verdict"]
    retry_count = state["retry_count"]

    print("\n" + "=" * 60)
    print("ROUTING")
    print("=" * 60)

    print(f"Verdict     : {verdict}")
    print(f"Retry count : {retry_count}")

    # 1. Good answer → finish
    if verdict == "GROUNDED":
        print("Decision    : OUTPUT GUARDRAIL")
        return "end"

    # 2. No answer → give retrieval ONE more chance
    if verdict == "ABSTAINED":

        if retry_count < 1:
            print("Decision    : RETRY")
            return "retry"

        print("Decision    : FALLBACK")
        return "fallback"

    # 3. Unsupported/wrong answer → allow TWO retries
    if verdict == "NOT_GROUNDED":

        if retry_count < 2:
            print("Decision    : RETRY")
            return "retry"

        print("Decision    : FALLBACK")
        return "fallback"

    # Safety case for unexpected verdict
    print("Decision    : FALLBACK")
    return "fallback"
# ============================================================
# REFORMULATE NODE
# ============================================================
def reformulate_node(state: RAGState):

    print("\n" + "=" * 60)
    print("REFORMULATE NODE")
    print("=" * 60)

    question = state["sanitized_question"]

    reason = state.get(
        "reason",
        "The previous answer was not sufficiently grounded in the retrieved context."
    )

    context = state.get(
        "context",
        ""
    )

    start_time = time.perf_counter()

    system_prompt = """
You are a retrieval-query optimizer for a RAG system.

Your task is to rewrite the user's original question into
a better keyword-rich search query after a failed retrieval
or generation attempt.

The improved query will be used by both:
- semantic vector search
- BM25 keyword search

Rules:

1. Preserve the original question's intent.

2. Do NOT answer the question.

3. Keep the important keywords from the original question.

4. You MAY add useful alternative terminology, synonyms,
   abbreviations, or domain-specific terms ONLY when those
   terms explicitly appear in the retrieved context.

5. Do NOT introduce factual claims or outside knowledge.

6. If the retrieved context contains terminology that appears
   related to the user's requested concept, include that
   terminology as additional search keywords.

7. If the critic reason identifies a missing concept, use it
   only when it helps retrieval and does not change the
   original intent.

8. Prefer keyword-rich retrieval language rather than a
   conversational sentence.

9. Keep the query concise.

10. Return ONLY the improved retrieval query.
"""

    user_prompt = f"""
ORIGINAL QUESTION:
{question}

CRITIC REASON:
{reason}

PREVIOUSLY RETRIEVED CONTEXT:
{context}

Rewrite the original question into a concise,
keyword-rich retrieval query.

Use useful terminology from the retrieved context when
it may improve retrieval.

Do not answer the question.
Do not use outside knowledge.
"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0,
        reasoning_effort="low"
    )

    new_question = (
        response
        .choices[0]
        .message
        .content
        .strip()
    )

    elapsed = (
        time.perf_counter()
        - start_time
    )

    print("\nOriginal question:")
    print(question)

    print("\nReformulated query:")
    print(new_question)

    print(
        f"\nReformulation time: "
        f"{elapsed:.4f} sec"
    )

    return {
        "retrieval_query": new_question,
        "retry_count": (
            state.get(
                "retry_count",
                0
            )
            + 1
        )
    }
# ============================================================
# FALLBACK NODE
# ============================================================

def fallback_node(state: RAGState):

    print("\n" + "=" * 60)
    print("FALLBACK NODE")
    print("=" * 60)

    fallback_answer = (
        "I couldn't find sufficient information in the "
        "provided documents to answer this question reliably."
    )

    print("\nFinal safe response:")
    print(fallback_answer)

    return {
        "answer": fallback_answer
    }
# ============================================================
# BUILD GRAPH
# ============================================================

builder = StateGraph(RAGState)

# input guaradrail node
builder.add_node("input_guardrail", input_guardrail_node)

# input block node
builder.add_node("blocked_input", blocked_input_node)


# Add retrieve node
builder.add_node(
    "retrieve",
    retrieve_node
)

# Add generate node
builder.add_node(
    "generate",
    generate_node
)

# Add critic node
builder.add_node(
    "critic",
    critic_node
)
# output guardrail node
builder.add_node("output_guardrail", output_guardrail_node)

# policy engine node
builder.add_node("policy_engine", policy_node)

# Add reformulate node
builder.add_node(
    "reformulate",
    reformulate_node
)

builder.add_node(
    "fallback",
    fallback_node
)
# START → input guardrail
builder.add_edge(START, "input_guardrail")

builder.add_conditional_edges(
    "input_guardrail",
    route_after_input_guardrail,
    {
        "allow": "retrieve",
        "block": "blocked_input"
    }
)

builder.add_edge("blocked_input", END)

# RETRIEVE → GENERATE
builder.add_edge(
    "retrieve",
    "generate"

)

# GENERATE -> CRITIC
builder.add_edge(
    "generate",
    "critic"
)

builder.add_conditional_edges(
    "critic",
    route_after_critic,
    {
        "end": "output_guardrail",
        "retry": "reformulate",
        "fallback": "fallback"
    }
)
builder.add_edge(
    "output_guardrail",
    "policy_engine"
)

builder.add_edge("policy_engine", END)

builder.add_edge(
    "reformulate",
    "retrieve"
)
builder.add_edge(
    "fallback",
    END
)

# Compile graph
graph = builder.compile()


