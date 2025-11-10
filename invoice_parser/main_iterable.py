#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Iterable, schema-validated extraction with self-consistency + critique.

Usage (examples):
  python main_iterable.py --invoice invoice.txt --schema schema.json
  cat invoice.txt | python main_iterable.py --schema schema.json

Env knobs (override defaults):
  NUM_SAMPLES=3
  MAX_PASSES=3
  ACCEPT_SCORE=0.90
  MAX_CRITIQUE_PASSES=1
  MODEL_NAME="gpt-4o-mini" (or any you wire in)
  TEMPERATURE=0.0

Requires:
  pip install jsonschema rapidfuzz (rapidfuzz optional but recommended)
"""

from __future__ import annotations
import os, sys, json, re, math, argparse, time, textwrap
from typing import Any, Dict, List, Tuple, Optional

# --- Optional deps ---
try:
    from rapidfuzz import fuzz
except Exception:
    fuzz = None  # we'll degrade gracefully

from jsonschema import validate
from jsonschema import ValidationError


# ========================
# LLM ADAPTER (PLUG YOURS)
# ========================

def llm(prompt: str, *, system: Optional[str] = None, temperature: float = 0.0,
        model: Optional[str] = None, max_tokens: Optional[int] = None) -> str:
    """
    Return a string response from your favorite model.
    Implement exactly one of the adapters below and keep the signature.

    By default this function raises until you wire it up.
    """
    # ---------- OPENAI PYTHON SDK (v1) EXAMPLE ----------
    # from openai import OpenAI
    # client = OpenAI()
    # msgs = []
    # if system:
    #     msgs.append({"role": "system", "content": system})
    # msgs.append({"role": "user", "content": prompt})
    # resp = client.chat.completions.create(
    #     model=model or os.getenv("MODEL_NAME", "gpt-4o-mini"),
    #     messages=msgs,
    #     temperature=temperature,
    #     max_tokens=max_tokens,
    # )
    # return resp.choices[0].message.content

    # ---------- ANTHROPIC PYTHON SDK EXAMPLE ----------
    # import anthropic
    # client = anthropic.Anthropic()
    # msgs = []
    # if system:
    #     msgs.append({"role": "system", "content": system})
    # # Claude uses a different interface; we flatten messages into a single string:
    # sys_txt = f"System:\n{system}\n\n" if system else ""
    # prompt_txt = sys_txt + f"User:\n{prompt}\n\nAssistant:"
    # resp = client.messages.create(
    #     model=model or os.getenv("MODEL_NAME", "claude-3-5-sonnet-latest"),
    #     max_tokens=max_tokens or 2000,
    #     temperature=temperature,
    #     messages=[{"role": "user", "content": prompt_txt}],
    # )
    # return "".join([blk.text for blk in resp.content if getattr(blk, 'type', '') == 'text'])

    raise RuntimeError(
        "Please wire the `llm()` function to your provider (see adapters in code)."
    )


# =================
# PROMPT TEMPLATES
# =================

GEN_SYSTEM = """You are a careful information extraction model.
- Output ONLY valid JSON matching the provided JSON Schema.
- Never include comments or explanations.
- Use null when uncertain. Do not hallucinate.
- All numbers must be numeric (no currency symbols).
"""

GEN_USER_TMPL = """Extract a JSON object from the following document text that strictly conforms to the given JSON Schema.

JSON Schema:
{schema}
Document
{doc}
Output ONLY the JSON object, nothing else.
"""

CRITIQUE_SYSTEM = """You are a QA verifier for invoice extraction.
Be strict and short. Output a compact JSON {{"valid": bool, "action": "accept"|"fix"|"regenerate"|"abstain", "comments": str, "patch": object|null}}.
- "fix" is for small, local corrections (e.g., rounding, obvious field typos).
- "regenerate" when structure is wrong or values contradict evidence.
- "abstain" if the document is too ambiguous.
- Include a minimal "patch" (a partial JSON) ONLY when action="fix".
"""

CRITIQUE_USER_TMPL = """Given the original document and a proposed extraction JSON, validate correctness, numeric invariants, evidence alignment (values visible in text), and date logic.

Document (start):
{doc_head}
Document (end):
{doc_tail}
Proposed JSON:
{candidate}


Respond ONLY with a single JSON object as specified.
"""


# ==================
# JSON UTILITIES
# ==================

JSON_BLOCK_RE = re.compile(r"\{(?:[^{}]|(?R))*\}", re.DOTALL)

def extract_first_json_block(text: str) -> Optional[str]:
    """Best-effort: find the first top-level JSON object."""
    # Simple heuristic: look for first '{' and try to balance.
    # Fallback to regex-of-objects.
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    m = JSON_BLOCK_RE.search(text)
    return m.group(0) if m else None

def parse_json_safely(text: str) -> Optional[dict]:
    block = extract_first_json_block(text)
    if not block:
        return None
    try:
        return json.loads(block)
    except Exception:
        # Try to fix common trailing commas, control chars, etc.
        cleaned = re.sub(r",\s*([}\]])", r"\1", block)
        try:
            return json.loads(cleaned)
        except Exception:
            return None


# ==================
# SCORING / CHECKS
# ==================

IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b", re.I)
DATE_RE = re.compile(r"\b(20\d{2}|19\d{2})[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])\b")
CURRENCY_CODE_RE = re.compile(r"\b(USD|EUR|GBP|CZK|PLN|CHF|AUD|CAD|JPY|SEK|NOK|DKK)\b", re.I)

def str_present(hay: str, needle: Any) -> bool:
    if not needle:
        return False
    s = str(needle).strip()
    if not s:
        return False
    return re.search(re.escape(s), hay, re.I) is not None

def fuzzy_present(hay: str, needle: Any, threshold: int = 85) -> bool:
    if not needle or not fuzz:
        return str_present(hay, needle)
    s = str(needle)
    return fuzz.partial_ratio(s.lower(), hay.lower()) >= threshold

def numeric(value) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None

def score_candidate(invoice_text: str, payload: dict) -> float:
    """
    Composite score 0..1:
      - structure/schema assumed validated earlier (+1)
      - presence of key fields (+3)
      - numeric invariants (+2)
      - evidence alignment (+2)
      - dates logic (+1)
      total max = 9
    """
    pts, max_pts = 0, 9
    # structure
    pts += 1

    # presence
    for k in ["invoice_number", "issue_date", "total_amount"]:
        if payload.get(k) not in (None, "", [], {}):
            pts += 1

    # numeric invariants (subtotal + VAT ~= total)
    try:
        lines = payload.get("items", []) or []
        subtotal = sum([(numeric(l.get("quantity")) or 0) * (numeric(l.get("unit_price")) or 0) for l in lines])
        vat_rate = numeric(payload.get("vat_rate")) or 0.0
        expected_total = round(subtotal * (1 + vat_rate), 2)
        got_total = round(numeric(payload.get("total_amount")) or 0.0, 2)
        if math.isfinite(expected_total) and abs(expected_total - got_total) <= 0.02:
            pts += 2
    except Exception:
        pass

    # evidence alignment
    ev = 0
    for k in ["supplier_name", "buyer_name", "currency"]:
        if fuzzy_present(invoice_text, payload.get(k)):
            ev += 1
    pts += min(ev, 2)

    # dates logic: issue_date <= due_date (when both present)
    issue = payload.get("issue_date")
    due = payload.get("due_date")
    if issue and due:
        # heuristic: lexicographic OK for ISO-ish formats; else just reward presence
        pts += 1 if str(issue) <= str(due) else 0
    else:
        pts += 0  # neither penalize nor reward missing dates here

    return max(0.0, min(1.0, pts / max_pts))


# =========================
# GENERATION & CRITIQUE
# =========================

def generate_json_once(doc: str, schema_text: str, *, temperature: float = 0.0,
                       model: Optional[str] = None) -> Optional[dict]:
    prompt = GEN_USER_TMPL.format(schema_text=schema_text, doc=doc)
    raw = llm(prompt, system=GEN_SYSTEM, temperature=temperature, model=model, max_tokens=None)
    return parse_json_safely(raw)

def generate_json_with_retries(doc: str, schema_text: str, attempts: int = 3,
                               temperature: float = 0.0, model: Optional[str] = None) -> Tuple[Optional[dict], str]:
    last_raw = ""
    for _ in range(max(1, attempts)):
        raw = llm(GEN_USER_TMPL.format(schema_text=schema_text, doc=doc),
                  system=GEN_SYSTEM, temperature=temperature, model=model)
        last_raw = raw
        parsed = parse_json_safely(raw)
        if parsed is not None:
            return parsed, raw
    return None, last_raw

def critique_json(doc: str, candidate: dict, *, model: Optional[str] = None) -> dict:
    head = doc[:1500]
    tail = doc[-1500:] if len(doc) > 1500 else ""
    prompt = CRITIQUE_USER_TMPL.format(doc_head=head, doc_tail=tail, candidate=json.dumps(candidate, ensure_ascii=False))
    raw = llm(prompt, system=CRITIQUE_SYSTEM, temperature=0.0, model=model)
    out = parse_json_safely(raw)
    if not isinstance(out, dict):
        return {"valid": False, "action": "regenerate", "comments": "Critique parser failed", "patch": None}
    # sanitize action
    action = out.get("action", "regenerate")
    if action not in {"accept", "fix", "regenerate", "abstain"}:
        action = "regenerate"
    out["action"] = action
    return out

def apply_fix_or_regenerate(doc: str, best: dict, crit: dict, *, schema_text: str,
                            model: Optional[str] = None) -> dict:
    if crit.get("action") == "fix" and isinstance(crit.get("patch"), dict):
        fixed = dict(best)
        # shallow merge
        for k, v in crit["patch"].items():
            fixed[k] = v
        return fixed
    # regenerate
    regen, _ = generate_json_with_retries(doc, schema_text, attempts=2, temperature=0.2, model=model)
    return regen or best


# ======================
# ITERABLE ORCHESTRATOR
# ======================

def run_iterable(doc: str, schema_text: str, schema_obj: dict) -> dict:
    # knobs
    NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "3"))
    MAX_PASSES = int(os.getenv("MAX_PASSES", "3"))
    ACCEPT_SCORE = float(os.getenv("ACCEPT_SCORE", "0.90"))
    MAX_CRITIQUE_PASSES = int(os.getenv("MAX_CRITIQUE_PASSES", "1"))
    MODEL_NAME = os.getenv("MODEL_NAME", None)
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

    best_overall = None
    best_overall_score = -1.0

    for p in range(MAX_PASSES):
        # 1) sample candidates
        candidates: List[dict] = []
        for _ in range(NUM_SAMPLES):
            payload, _raw = generate_json_with_retries(
                doc, schema_text, attempts=2, temperature=TEMPERATURE, model=MODEL_NAME
            )
            if payload is not None:
                # schema validation
                try:
                    validate(instance=payload, schema=schema_obj)
                    candidates.append(payload)
                except ValidationError:
                    continue

        # fallback: try at least one
        if not candidates:
            payload, _ = generate_json_with_retries(doc, schema_text, attempts=3, temperature=TEMPERATURE, model=MODEL_NAME)
            if payload is None:
                continue
            try:
                validate(instance=payload, schema=schema_obj)
                candidates.append(payload)
            except ValidationError:
                # force minimal recovery: keep even if invalid, so critique may push to regen
                candidates.append(payload)

        # 2) select best by rule-based score
        scored = []
        for c in candidates:
            s = score_candidate(doc, c)
            scored.append((s, c))
        scored.sort(key=lambda t: t[0], reverse=True)
        best, score = scored[0]

        # Track global best
        if score > best_overall_score:
            best_overall, best_overall_score = best, score

        # 3) optional critique loop
        for _crit_i in range(MAX_CRITIQUE_PASSES):
            crit = critique_json(doc, best, model=MODEL_NAME)
            action = crit.get("action")
            if action == "accept":
                # Accept only if score good enough
                if score >= ACCEPT_SCORE:
                    return best
                else:
                    # try to nudge via small fix if patch exists
                    if isinstance(crit.get("patch"), dict):
                        best = apply_fix_or_regenerate(doc, best, crit, schema_text=schema_text, model=MODEL_NAME)
                        try:
                            validate(instance=best, schema=schema_obj)
                        except ValidationError:
                            pass
                        score = score_candidate(doc, best)
                        if score >= ACCEPT_SCORE:
                            return best
                    break  # done with critique
            elif action in {"fix", "regenerate"}:
                best = apply_fix_or_regenerate(doc, best, crit, schema_text=schema_text, model=MODEL_NAME)
                try:
                    validate(instance=best, schema=schema_obj)
                except ValidationError:
                    # keep going; a later pass may recover
                    pass
                score = score_candidate(doc, best)
                # continue to next critique pass (if any)
            else:  # abstain or unknown
                break

        # 4) early accept on score
        if score >= ACCEPT_SCORE:
            return best

        # loop again, carrying best_overall implicitly

    # After passes: return the best we got (even if not at threshold)
    return best_overall or {}


# ===============
# I/O + RUNNER
# ===============

def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def main():
    ap = argparse.ArgumentParser(description="Iterable extraction with schema + critique.")
    ap.add_argument("--invoice", type=str, help="Path to invoice/document text file. If omitted, reads STDIN.")
    ap.add_argument("--schema", type=str, required=True, help="Path to JSON Schema file.")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = ap.parse_args()

    # Read inputs
    doc = read_text_file(args.invoice) if args.invoice else sys.stdin.read()
    schema_text = read_text_file(args.schema)
    try:
        schema_obj = json.loads(schema_text)
    except Exception as e:
        print(f"[fatal] schema JSON parse error: {e}", file=sys.stderr)
        sys.exit(2)

    # Run
    try:
        result = run_iterable(doc, schema_text, schema_obj)
        if not result:
            print(json.dumps({"_error": "no_result"}, ensure_ascii=False))
            return
        # Final sanity: validate (if fails, still print but mark _invalid)
        valid = True
        try:
            validate(instance=result, schema=schema_obj)
        except ValidationError as ve:
            valid = False
            result = {"_invalid": True, "_reason": str(ve)[:500], "data": result}
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
    except RuntimeError as rt:
        # likely the LLM adapter not wired
        print(f"[fatal] {rt}", file=sys.stderr)
        sys.exit(3)

if __name__ == "__main__":
    main()

