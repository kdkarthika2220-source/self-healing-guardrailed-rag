import os
import re

from groq import Groq
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from pydantic import BaseModel, Field, ValidationError


GROQ_MODEL = "openai/gpt-oss-120b"

groq_client = None


def get_groq_client():

    global groq_client

    if groq_client is None:

        groq_client = Groq(
            api_key=os.environ["GROQ_API_KEY"]
        )

    return groq_client


# ============================================================
# OUTPUT SCHEMAS
# ============================================================

class RAGOutput(BaseModel):

    answer: str = Field(
        min_length=1,
        description="Final grounded answer returned to the user"
    )

    safe: bool = Field(
        description="Whether the answer passed output safety checks"
    )


class OutputSafetyResult(BaseModel):

    safe: bool = Field(
        description="Whether the generated answer is safe to return"
    )

    reason: str = Field(
        description="Short reason for the safety decision"
    )


# ============================================================
# OUTPUT SAFETY
# ============================================================

def check_output_safety(answer: str) -> dict:

    system_prompt = """
You are an output safety classifier for a RAG system.

Evaluate the generated answer.

Mark SAFE only if:

1. The answer is not toxic, abusive, hateful, or threatening.
2. The answer does not provide dangerous or harmful instructions.
3. The answer is appropriate for a professional question-answering system.

Do NOT evaluate factual grounding here.
Grounding is checked by another critic.

Return only JSON.
"""
    client = get_groq_client()

    response = client.chat.completions.create(
    
        model=GROQ_MODEL,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"ANSWER:\n{answer}"
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "output_safety_result",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "safe": {
                            "type": "boolean"
                        },
                        "reason": {
                            "type": "string"
                        }
                    },
                    "required": [
                        "safe",
                        "reason"
                    ],
                    "additionalProperties": False
                }
            }
        }
    )

    raw_result = response.choices[0].message.content

    result = OutputSafetyResult.model_validate_json(
        raw_result
    )

    return {
        "safe": result.safe,
        "reason": result.reason
    }


# ============================================================
# PROMPT INJECTION
# ============================================================

BLOCK_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all previous instructions",
    r"forget previous instructions",
    r"reveal your system prompt",
    r"show your system prompt",
    r"bypass.*instructions",
    r"override.*instructions",
    r"jailbreak",
]


def detect_prompt_injection(user_input: str) -> dict:

    normalized_input = user_input.lower().strip()

    for pattern in BLOCK_PATTERNS:

        if re.search(pattern, normalized_input):

            return {
                "is_safe": False,
                "reason": (
                    f"Prompt injection pattern detected: {pattern}"
                )
            }

    return {
        "is_safe": True,
        "reason": "No known prompt injection pattern detected."
    }


# ============================================================
# PII REDACTION lazy load
# ============================================================

pii_analyzer = None
pii_anonymizer = None


def get_pii_analyzer():

    global pii_analyzer

    if pii_analyzer is None:

        configuration = {
            "nlp_engine_name": "spacy",
            "models": [
                {
                    "lang_code": "en",
                    "model_name": "en_core_web_sm"
                }
            ]
        }

        provider = NlpEngineProvider(
            nlp_configuration=configuration
        )

        nlp_engine = provider.create_engine()

        pii_analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=["en"]
        )

    return pii_analyzer


def get_pii_anonymizer():

    global pii_anonymizer

    if pii_anonymizer is None:

        pii_anonymizer = AnonymizerEngine()

    return pii_anonymizer

def detect_and_redact_pii(user_input: str) -> dict:

    analyzer = get_pii_analyzer()
    anonymizer = get_pii_anonymizer()

    results = analyzer.analyze(
        text=user_input,
        language="en",
        entities=[
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD"
        ]
    )

    if not results:

        return {
            "pii_detected": False,
            "redacted_text": user_input,
            "entities": []
        }

    anonymized = anonymizer.anonymize(
        text=user_input,
        analyzer_results=results
    )

    entities = list({
        result.entity_type
        for result in results
    })

    return {
        "pii_detected": True,
        "redacted_text": anonymized.text,
        "entities": entities
    }


# ============================================================
# OUTPUT SCHEMA VALIDATION
# ============================================================

def validate_rag_output(answer: str) -> dict:

    try:

        validated = RAGOutput(
            answer=answer,
            safe=True
        )

        return {
            "valid": True,
            "answer": validated.answer,
            "reason": "Output schema validation passed."
        }

    except ValidationError as error:

        return {
            "valid": False,
            "answer": "",
            "reason": (
                f"Output schema validation failed: {error}"
            )
        }