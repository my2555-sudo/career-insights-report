from __future__ import annotations
import os, time, json
from pathlib import Path
from typing import List, Dict, Any, Iterable

import pandas as pd
import numpy as np
import requests
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv

load_dotenv(override=False)

# ===== Config =====
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")

REQUEST_TIMEOUT = 60
MAX_TOKENS = 700
SLEEP_BETWEEN_CALLS = float(os.getenv("DEEPSEEK_SLEEP", "0.6"))

# Output dirs (used by main.py too)
ENRICHED_DIR = Path("data/enriched")
ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = Path("examples") 
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# Buckets we ask the model to use
TOOLKIT_BUCKETS = [
    "Programming", "SQL & Warehouses", "Data Engineering", "ML/AI",
    "Visualization", "Cloud", "MLOps & Infra", "APIs & Services",
    "Analytics/Experimentation", "Soft Skills", "Other"
]

SYSTEM_PROMPT = (
    "You are a concise, practical career coach for data/tech roles. "
    "Read the provided market signals and the JD text. "
    "Return a short, actionable output in strict JSON."
)

def _mk_user_prompt(row: pd.Series) -> str:
    jt = row.get("job_title", "")
    edu = row.get("education_level", "")
    yr = row.get("years_of_experience", "")
    sal = row.get("salary", "")
    mean_t = row.get("salary_mean_title", np.nan)
    mean_te = row.get("salary_mean_titleedu", np.nan)
    d_title = row.get("delta_vs_title_mean", np.nan)
    d_te = row.get("delta_vs_titleedu_mean", np.nan)

    jd_corpus = row.get("jd_corpus", "")
    if isinstance(jd_corpus, str) and len(jd_corpus) > 7000:
        jd_corpus = jd_corpus[:7000] + "\n...[truncated]"
    jd_skills = row.get("jd_skills", "")

    buckets = ", ".join(TOOLKIT_BUCKETS)

    return "\n".join([
        f"Role: {jt}",
        f"Education: {edu}",
        f"Experience (years): {yr}",
        f"Current Salary (USD): {sal}",
        f"Title mean salary: {mean_t}",
        f"Title+Education mean salary: {mean_te}",
        f"Delta vs title mean (ratio): {d_title}",
        f"Delta vs title+education mean (ratio): {d_te}",
        "",
        "You are given recent JD text for this role. Extract concrete tools/skills and categorize them into:",
        buckets,
        "",
        "JD TEXT:",
        jd_corpus or "(no JD text available)",
        "",
        "Side-signal (regex-mined keywords; optional):",
        jd_skills or "(none)",
        "",
        "Return STRICT JSON with keys:",
        "- pay_label (string)",
        "- top_gaps (list of strings)",
        "- plan (string, 2-3 sentences)",
        "- counterfactual_salary (number or null)",
        "- skills_by_toolkit (object: toolkit -> unique list of normalized skills, <=10 each)",
        "",
        "Normalization rules:",
        " - Deduplicate synonyms (e.g., 'tf' -> 'TensorFlow').",
        " - Prefer canonical names (e.g., 'Power BI', 'AWS', 'PyTorch', 'Airflow').",
    ])

def _headers() -> Dict[str, str]:
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("Missing DEEPSEEK_API_KEY in environment/.env")
    return {"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"}

@retry(wait=wait_exponential(multiplier=1, min=1, max=20), stop=stop_after_attempt(3))
def call_deepseek_chat(messages: List[Dict[str, str]], model: str = DEFAULT_MODEL, max_tokens: int = MAX_TOKENS) -> str:
    url = f"{DEEPSEEK_BASE_URL}/chat/completions"
    payload = {"model": model, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens}
    resp = requests.post(url, headers=_headers(), json=payload, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data, ensure_ascii=False)

def _safe_json_parse(s: str) -> Dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        try:
            start = s.find("{"); end = s.rfind("}")
            if start != -1 and end != -1 and end > start:
                return json.loads(s[start:end+1])
        except Exception:
            pass
    return {
        "pay_label": "Unknown",
        "top_gaps": [],
        "plan": s[:800],
        "counterfactual_salary": None,
        "skills_by_toolkit": {},
    }

def rule_based_enrich(row: pd.Series) -> Dict[str, Any]:
    mean_t = row.get("salary_mean_title", np.nan)
    sal = row.get("salary", np.nan)
    d = row.get("delta_vs_title_mean", np.nan)

    def _pay_label():
        if pd.isna(d): return "Unknown"
        if d <= -0.15: return "Well below market (~15%↓)"
        if d <= -0.05: return "Slightly below market (~5–15%↓)"
        if d < 0.05:  return "Around market (±5%)"
        if d < 0.20:  return "Above market (~5–20%↑)"
        return "Well above market (>20%↑)"

    cf = None
    if pd.notna(mean_t) and pd.notna(sal):
        gap = float(mean_t) - float(sal)
        cf = float(sal) + 0.6 * gap

    return {
        "pay_label": _pay_label(),
        "top_gaps": ["SQL", "Python", "Experimentation/AB"],
        "plan": "Strengthen SQL and Python; deliver two end-to-end analytics projects; target Senior-level in 12–24 months.",
        "counterfactual_salary": round(cf, 0) if cf else None,
        "skills_by_toolkit": {},
    }

def _normalize_bucketed(sbt: Any) -> Dict[str, List[str]]:
    if not isinstance(sbt, dict):
        return {}
    cleaned: Dict[str, List[str]] = {}
    for bucket, skills in sbt.items():
        if not isinstance(skills, list):
            continue
        seen, out = set(), []
        for s in skills:
            s = str(s or "").strip()
            if not s:
                continue
            disp = s.upper() if len(s) <= 4 else s.title()
            if disp.lower() == "Power Bi":
                disp = "Power BI"
            if disp not in seen:
                seen.add(disp)
                out.append(disp)
            if len(out) >= 10:
                break
        if out:
            cleaned[str(bucket)] = out
    return cleaned

def enrich_rows(rows: Iterable[pd.Series], model: str = DEFAULT_MODEL, use_api: bool = True) -> List[Dict[str, Any]]:
    """
    For each row (expects fields used in _mk_user_prompt), call DeepSeek to:
      - read JD text and produce guidance
      - extract skills_by_toolkit
    Merge with a rule-based fallback; normalize types; return list of dicts.
    """
    results: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, pd.Series):
            try:
                r = pd.Series(r)
            except Exception:
                r = pd.Series({})

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _mk_user_prompt(r)},
        ]

        out = rule_based_enrich(r)

        if use_api:
            try:
                content = call_deepseek_chat(messages, model=model, max_tokens=MAX_TOKENS)
                parsed = _safe_json_parse(content)
                # Model wins when non-empty
                for k, v in parsed.items():
                    if v not in (None, "", []):
                        out[k] = v
                time.sleep(SLEEP_BETWEEN_CALLS)
            except Exception as e:
                out["api_error"] = str(e)

        # normalize
        if isinstance(out.get("top_gaps"), str):
            out["top_gaps"] = [out["top_gaps"]]
        try:
            if out.get("counterfactual_salary") is not None:
                out["counterfactual_salary"] = float(out["counterfactual_salary"])
        except Exception:
            out["counterfactual_salary"] = None

        out["skills_by_toolkit"] = _normalize_bucketed(out.get("skills_by_toolkit"))

        # convenience long-form (not used by main, but handy)
        flat_rows = []
        for bucket, skills in out["skills_by_toolkit"].items():
            for sk in skills:
                flat_rows.append({"toolkit": bucket, "skill": sk})
        out["skills_bucketed"] = flat_rows

        results.append(out)
    return results
