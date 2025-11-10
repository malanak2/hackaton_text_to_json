import os
import re
import json
from typing import Tuple, Optional
from jsonschema import validate, ValidationError

# (Optional) Remove this line in real use; it's just for local testing.
os.environ["HF_TOKEN"] = "tady_vlozit_huggingface_token_anebo_env"

# =========================
# CONFIG
# =========================
HF_TOKEN = os.getenv("HF_TOKEN")  # set via env; avoid hardcoding
if not HF_TOKEN:
    raise RuntimeError(
        "HF_TOKEN env var not set. Create one at https://huggingface.co/settings/tokens and export HF_TOKEN."
    )

# Choose a model available via Hugging Face OpenAI-compatible router
HF_MODEL = os.getenv("HF_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct:novita")

INVOICE_FILE = os.getenv("INVOICE_FILE", "invoice.txt")
SCHEMA_FILE = os.getenv("SCHEMA_FILE", "schema_faktur_ciste.json")
OUTPUT_JSON = os.getenv("OUTPUT_JSON", "parsed_invoice.json")

# Generation parameters
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "768"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))
MAX_JSON_RETRIES = int(os.getenv("MAX_JSON_RETRIES", "3"))
MAX_CRITIQUE_PASSES = int(os.getenv("MAX_CRITIQUE_PASSES", "0"))
MAX_CRITIQUE_RETRIES = int(os.getenv("MAX_CRITIQUE_RETRIES", "0"))

# =========================
# OpenAI client (points to HF Router)
# =========================
from openai import OpenAI
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

# -------------------------
# Helpers
# -------------------------

def clean_text(txt: str) -> str:
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def chat_once(prompt: str, model: str = HF_MODEL) -> str:
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful, deterministic JSON-producing assistant. "
                    "When asked for JSON, respond with a single valid JSON object and nothing else."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_NEW_TOKENS,
    )
    return completion.choices[0].message.content


def extract_first_json_block(text: str) -> Optional[str]:
    """Extract the first top-level JSON object from arbitrary text."""
    match = re.search(r"\{.*\}", text, re.S)
    return match.group(0) if match else None


def json_only_prompt(invoice_text: str, schema_text: str) -> str:
    """
    Build a prompt that instructs the model to output JSON conforming to the provided JSON Schema.
    The schema is loaded from file (SCHEMA_FILE) elsewhere and passed in as schema_text.
    """
    return f"""
You are an expert invoice parser. Analyze the following invoice text and extract key fields.

Output requirements:
- Return ONLY a single valid JSON object (no code fences, no extra prose).
- The JSON MUST strictly validate against the following JSON Schema (Draft 7/2020 compatible):

SCHEMA:
{schema_text}
If a field is missing/uncertain in the invoice, set it to null (while still conforming to the schema). 
DO NOT HALLUCINATE!

Invoice text:
{invoice_text}
""".strip()


def generate_json_with_retries(invoice_text: str, schema_text: str, max_attempts: int = MAX_JSON_RETRIES) -> Tuple[dict, str]:
    """Try up to max_attempts to get valid JSON. Returns (payload, raw_model_text)."""
    last_raw = ""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        raw = chat_once(json_only_prompt(invoice_text, schema_text))
        last_raw = raw
        json_str = extract_first_json_block(raw)
        if not json_str:
            # Ask model to fix formatting explicitly (re-provide schema to anchor structure)
            repair_prompt = (
                "Your previous output did not contain a parsable JSON object. "
                "Re-output the same content as a single valid JSON object with no extra text. "
                "Ensure the JSON validates against the provided schema below.\n\n"
                f"SCHEMA:\n```\n{schema_text}\n```"
            )
            raw = chat_once(repair_prompt)
            json_str = extract_first_json_block(raw)
        try:
            if json_str is None:
                raise ValueError("No JSON detected in model output.")
            payload = json.loads(json_str)
            return payload, last_raw
        except Exception as e:
            last_error = e
    # If we reach here, all attempts failed
    raise ValueError(f"Failed to produce valid JSON after {max_attempts} attempts. Last error: {last_error}\nLast raw: {last_raw}")


def validate_against_schema(payload: dict, schema_path: str) -> None:
    with open(schema_path, "r", encoding="utf-8") as s:
        schema = json.load(s)
    validate(instance=payload, schema=schema)


def critique_prompt(invoice_text: str, json_payload: dict) -> str:
    return (
        "You are an exacting auditor called 'LLM critique'.\n"
        "Task: strictly assess whether the JSON extraction correctly reflects the invoice text.\n"
        "Rules:\n"
        "- Check for numerical mismatches (totals, VAT, currency).\n"
        "- Check dates, invoice number, supplier identity.\n"
        "- Ensure items list matches text (descriptions, quantities, unit prices, VAT rates).\n"
        "- If any field is uncertain or missing in the text, JSON should set it to null.\n"
        "Output a single JSON object with fields: {\n"
        "  \"valid\": boolean,\n"
        "  \"issues\": [string],\n"
        "  \"severity\": \"low\"|\"medium\"|\"high\",\n"
        "  \"action\": \"accept\"|\"fix\"|\"regenerate\",\n"
        "  \"suggested_fix\": string\n"
        "}. No explanations outside JSON.\n\n"
        f"Invoice text:\n```\n{invoice_text}\n```\n\n"
        f"Candidate JSON:\n```\n{json.dumps(json_payload, ensure_ascii=False, indent=2)}\n```\n"
    )


def critique_json(invoice_text: str, json_payload: dict, max_attempts: int = MAX_CRITIQUE_RETRIES) -> dict:
    """Critique with retries: ensure we get valid JSON back from the model.
    On parse failure, we ask the model to re-output JSON-only and retry up to max_attempts.
    """
    last_raw = ""
    last_error: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        raw = chat_once(critique_prompt(invoice_text, json_payload))
        last_raw = raw
        crit_str = extract_first_json_block(raw)
        if not crit_str:
            # Ask to re-output JSON only
            repair_prompt = (
                "Your previous critique did not contain a parsable JSON object. "
                "Re-output the same critique as a single valid JSON object with no extra text."
            )
            raw = chat_once(repair_prompt)
            last_raw = raw
            crit_str = extract_first_json_block(raw)
        try:
            if crit_str is None:
                raise ValueError("Critique step did not return JSON.")
            return json.loads(crit_str)
        except Exception as e:
            last_error = e
    raise ValueError(f"Failed to get valid critique JSON after {max_attempts} attempts: {last_error} Last raw: {last_raw}")


def apply_fix_or_regenerate(invoice_text: str, json_payload: dict, critique: dict) -> dict:
    """Use the critique to ask the model to fix or regenerate JSON, then return the new JSON."""
    action = critique.get("action", "fix")
    guidance = critique.get("suggested_fix", "")

    if action not in {"fix", "regenerate"}:
        # Default to fix when unsure
        action = "fix"

    if action == "fix":
        prompt = (
            "You previously produced a JSON extraction for an invoice. "
            "Apply the following critique to correct mistakes and output a single valid JSON object ONLY.\n\n"
            f"Critique guidance:\n{json.dumps(critique, ensure_ascii=False)}\n\n"
            f"Original invoice text:\n```\n{invoice_text}\n```\n\n"
            f"Current JSON to correct:\n```\n{json.dumps(json_payload, ensure_ascii=False)}\n```\n"
        )
    else:  # regenerate
        prompt = (
            "Regenerate the invoice JSON extraction from scratch using the rules, "
            "ensuring it matches the invoice text precisely. Output a single valid JSON object ONLY.\n\n"
            f"Critique guidance:\n{json.dumps(critique, ensure_ascii=False)}\n\n"
            f"Invoice text:\n```\n{invoice_text}\n```\n"
        )

    raw = chat_once(prompt)
    json_str = extract_first_json_block(raw)
    if not json_str:
        raise ValueError("Repair/regenerate step did not return JSON.")
    return json.loads(json_str)


# -------------------------
# Pipeline
# -------------------------
import json
def run_pipeline(txt_b) -> dict:
    # 1) Demo quick usage (sanity check against the API)
    demo = client.chat.completions.create(
        model=HF_MODEL,
        messages=[{"role": "user", "content": "What is the capital of France?"}],
        temperature=TEMPERATURE,
        max_tokens=64,
    )
    print("Demo (capital of France) →", demo.choices[0].message.content, ", this mean that the API of a chatbot is reachable.")

    txt = txt_b.decode("utf-8")
    invoice_text = clean_text(txt)

    # 3) Load the schema text once (for prompting) and object (for validation)
    with open(SCHEMA_FILE, "r", encoding="utf-8") as s:
        schema_text = s.read()
        schema_obj = json.loads(schema_text)

    # 4) Generate structured JSON with retries (schema-aware prompting)
    payload, raw_text = generate_json_with_retries(invoice_text, schema_text, MAX_JSON_RETRIES)

    # 5) Validate against JSON Schema (schema-level verification)
    try:
        validate(instance=payload, schema=schema_obj)
        print("✅ JSON passes schema validation on first pass.")
    except ValidationError as e:
        print("⚠️ Schema validation failed on first pass:", e.message)

    # 6) LLM Critique loop (semantic/business verification)
    for i in range(MAX_CRITIQUE_PASSES):
        crit = critique_json(invoice_text, payload, MAX_CRITIQUE_RETRIES)
        print(f"Critique pass {i+1}:", crit)
        if bool(crit.get("valid", False)) and crit.get("action", "accept") == "accept":
            print("✅ Critique accepted the JSON.")
            break
        # otherwise fix or regenerate
        payload = apply_fix_or_regenerate(invoice_text, payload, crit)
        # re-validate after fix/regenerate
        try:
            validate(instance=payload, schema=schema_obj)
            print("✅ JSON passes schema validation after fix/regenerate.")
        except ValidationError as e:
            print("⚠️ Schema validation error after fix/regenerate:", e.message)

    # 7) Save final JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as out:
        json.dump(payload, out, indent=2, ensure_ascii=False)
    print(f"Saved {OUTPUT_JSON}")

    return payload

from flask import Flask, request
app = Flask(__name__)
@app.route('/', methods=['POST'])
def parse_txt():
    print("Received request")
    data = request.data
    jsonD = json.dumps(run_pipeline(data))
    print(jsonD)
    return jsonD

if __name__ == "__main__":
    app.run()
