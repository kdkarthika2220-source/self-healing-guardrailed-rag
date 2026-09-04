import os
import time

from groq import Groq
from pydantic import BaseModel, Field


# ============================================================
# CONFIGURATION
# ============================================================

GROQ_MODEL = "openai/gpt-oss-120b"

groq_client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)


# ============================================================
# STRUCTURED OUTPUT SCHEMA
# ============================================================

class CriticResult(BaseModel):
    verdict: str = Field(
        description="Must be either GROUNDED or NOT_GROUNDED"
    )

    reason: str = Field(
        description="Short explanation for the verdict"
    )


# ============================================================
# CRITIC FUNCTION
# ============================================================

def critique_answer(question, context, answer):

    system_prompt = """
You are a strict RAG evaluator.

Determine whether the ANSWER is fully supported by the
provided CONTEXT.

Rules:

1. Use ONLY the provided context.
2. Do not use outside knowledge.
3. If the answer contains information not supported by
   the context, classify it as NOT_GROUNDED.
4. If the context does not contain enough information to
   answer the question, classify it as NOT_GROUNDED.
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

    start_time = time.perf_counter()

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

    elapsed = time.perf_counter() - start_time

    raw_result = response.choices[0].message.content

    # Convert JSON response into a validated Pydantic object
    critic_result = CriticResult.model_validate_json(raw_result)

    return critic_result, elapsed


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    question = "What is a digital health system?"

    context = """
    A digital health system comprises all of the digital technology
    used to support the operations of the overall health system,
    including software applications and systems, devices and
    hardware, technologies, and the underlying information
    infrastructure.
    """

    answer = """
    A digital health system comprises digital technologies used
    to support the operations of the overall health system,
    including software applications, devices, hardware,
    technologies, and information infrastructure.
    """

    print("=" * 60)
    print("STRUCTURED CRITIC TEST")
    print("=" * 60)

    print("\nQUESTION:")
    print(question)

    print("\nANSWER:")
    print(answer)

    print("\nCRITIC:")

    result, elapsed = critique_answer(
        question,
        context,
        answer
    )

    print("\nVerdict :", result.verdict)
    print("Reason  :", result.reason)

    print("\nCRITIC LATENCY:")
    print(f"{elapsed:.4f} sec")

    print("\nStructured critic test complete!")