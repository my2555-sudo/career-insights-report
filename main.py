# -*- coding: utf-8 -*-
"""
Interactive Career Insights CLI (English)
- Ask user questions interactively: age, education, job title, years, salary.
- If no clean data: auto-download Kaggle dataset and compute title stats (one-time).
- Probe The Muse on-demand for the given role (few pages, US/Remote), extract common skills.
- Produce two comparable reports with the same structure:
    1) Baseline (no LLM) — regex-mined skills + data benchmarks only
    2) DeepSeek (LLM) — reads JD text and returns guidance + bucketed skills
- Persist:
    - data/enriched/salary_enriched.csv (wide table, 1 row per run)
    - data/enriched/muse_skills_bucketed.csv (long table of toolkit-bucketed skills from DeepSeek)
    - examples/employee_baseline_<slug>.md
    - examples/employee_deepseek_<slug>.md
"""

from __future__ import annotations
import os, sys, json, re, shutil, argparse, logging
from pathlib import Path
import difflib
from typing import Optional, List, Dict

import pandas as pd
import numpy as np
import requests
import kagglehub
from tenacity import retry, stop_after_attempt, wait_exponential
from bs4 import BeautifulSoup

# Optional DeepSeek enrichment
try:
    import deepseek_enrichment as dse
except Exception:
    dse = None

# --------------------------
# Paths & constants
# --------------------------
RAW_DIR = Path("data/raw")
ENRICHED_DIR = Path("data/enriched")
EXAMPLES_DIR = Path("examples")
REPORTS_DIR = EXAMPLES_DIR


RAW_CSV_PATH = RAW_DIR / "salary.csv"
CLEAN_CSV_PATH = ENRICHED_DIR / "salary_clean.csv"
JOB_STATS_PATH = ENRICHED_DIR / "job_salary_stats.csv"
JOBEDU_STATS_PATH = ENRICHED_DIR / "jobedu_salary_stats.csv"

ENRICHED_OUT_PATH = ENRICHED_DIR / "salary_enriched.csv"
BUCKET_OUT_PATH = ENRICHED_DIR / "muse_skills_bucketed.csv"

# The Muse API
MUSE_BASE = "https://www.themuse.com/api/public/jobs"
DEFAULT_HEADERS = {"User-Agent": "career-insights/0.4 (+https://example.local)"}

US_STATES = {
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC"
}

EDU_MAP = {
    "high school": "High School",
    "associate": "Associate's", "associates": "Associate's",
    "bachelor": "Bachelor's", "bachelors": "Bachelor's",
    "master": "Master's", "masters": "Master's",
    "phd": "PhD", "doctorate": "PhD",
}

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("career-cli")


# --------------------------
# Helpers (I/O, validation)
# --------------------------
def normalize_text(x: str) -> str:
    return re.sub(r"\s+", " ", str(x).strip())

def norm_title(x: str) -> str:
    return normalize_text(x).lower()

def standardize_education(x: str) -> str:
    if not isinstance(x, str):
        return ""
    k = x.lower()
    for key, val in EDU_MAP.items():
        if key in k:
            return val
    return x.strip().title()

def _winsorize_series(s: pd.Series, lower=0.01, upper=0.99) -> pd.Series:
    if s.empty:
        return s
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lower=lo, upper=hi)

def ask(prompt: str, validate=None, default: Optional[str] = None):
    while True:
        raw = input(f"{prompt}{' ['+str(default)+']' if default is not None else ''}: ").strip()
        if not raw and default is not None:
            raw = str(default)
        if not validate:
            return raw
        ok, val_or_msg = validate(raw)
        if ok:
            return val_or_msg
        print(f"  ⚠️ {val_or_msg}")

def v_int_range(lo: int, hi: int):
    def _v(x: str):
        try:
            v = int(x)
            if lo <= v <= hi:
                return True, v
            return False, f"Please enter an integer in [{lo}–{hi}]."
        except:
            return False, "Please enter an integer."
    return _v

def v_float_range(lo: float, hi: float):
    def _v(x: str):
        try:
            v = float(x)
            if lo <= v <= hi:
                return True, v
            return False, f"Please enter a number in [{lo}–{hi}]."
        except:
            return False, "Please enter a number."
    return _v

def v_money_pos():
    def _v(x: str):
        x = x.replace(",", "")
        try:
            v = float(x)
            if v > 0:
                return True, v
            return False, "Salary must be > 0."
        except:
            return False, "Please enter a salary (commas allowed)."
    return _v

def v_nonempty():
    def _v(x: str):
        if x.strip():
            return True, x
        return False, "This cannot be empty."
    return _v

def fmt_money(x: float) -> str:
    try:
        return f"${float(x):,.0f}"
    except:
        return "N/A"

def percent(x: float) -> str:
    try:
        return f"{float(x)*100:.1f}%"
    except:
        return "N/A"

def safe_div(n, d):
    try:
        n = float(n); d = float(d)
        if d == 0: return np.nan
        return n / d
    except:
        return np.nan


# --------------------------
# Kaggle extract + clean (one-time)
# --------------------------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _get_json(url: str, params: dict | None = None, timeout: int = 30) -> dict:
    r = requests.get(url, params=params, timeout=timeout, headers=DEFAULT_HEADERS)
    r.raise_for_status()
    return r.json()

def download_kaggle_dataset(dataset_slug: str) -> Path:
    log.info(f"⬇️  Downloading Kaggle dataset: {dataset_slug}")
    path_str = kagglehub.dataset_download(dataset_slug)
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Kaggle dataset path not found: {path}")
    log.info(f"   Done: {path}")
    return path

def locate_source_file(kaggle_dir: Path, candidate_filenames: List[str]) -> Path:
    for name in candidate_filenames:
        f = kaggle_dir / name
        if f.exists():
            return f
    for f in kaggle_dir.rglob("*.csv"):
        if "salary" in f.name.lower():
            return f
    raise FileNotFoundError("Could not find a salary CSV inside the Kaggle dataset.")

def ensure_clean_data(dataset_slug="wardabilal/salary-prediction-dataset",
                      candidate_filenames=None,
                      force: bool=False):
    candidate_filenames = candidate_filenames or [
        "Salary Data.csv", "Salary_Data.csv", "salary.csv"
    ]
    if CLEAN_CSV_PATH.exists() and JOB_STATS_PATH.exists() and JOBEDU_STATS_PATH.exists() and not force:
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    kdir = download_kaggle_dataset(dataset_slug)
    src_csv = locate_source_file(kdir, candidate_filenames)
    shutil.copy(src_csv, RAW_CSV_PATH)

    df = pd.read_csv(RAW_CSV_PATH)
    lc = {c.lower().strip(): c for c in df.columns}
    colmap = {}
    def pick(keys, target):
        for k in keys:
            if k in lc:
                colmap[lc[k]] = target
                return
    pick(["age"], "age")
    pick(["gender"], "gender")
    pick(["education level","education_level","education"], "education_level")
    pick(["job title","job_title","title"], "job_title")
    pick(["years of experience","years_of_experience","experience","years"], "years_of_experience")
    pick(["salary","annual salary","annual_salary"], "salary")

    df = df.rename(columns=colmap)
    keep = [c for c in ["age","gender","education_level","job_title","years_of_experience","salary"] if c in df.columns]
    df = df[keep].copy()

    for c in ["gender","education_level","job_title"]:
        if c in df.columns:
            df[c] = df[c].astype(str).map(normalize_text)
    if "education_level" in df.columns:
        df["education_level"] = df["education_level"].map(standardize_education)
    if "age" in df.columns:
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
    if "years_of_experience" in df.columns:
        df["years_of_experience"] = pd.to_numeric(df["years_of_experience"], errors="coerce")
    if "salary" in df.columns:
        df["salary"] = pd.to_numeric(df["salary"], errors="coerce")
        df["salary"] = _winsorize_series(df["salary"], 0.01, 0.99)

    df = df.dropna(subset=["job_title","years_of_experience","salary"])
    df = df[(df["years_of_experience"] >= 0) & (df["years_of_experience"] <= 60)]
    df = df[df["salary"] > 0]
    df["job_title"] = df["job_title"].astype(str)
    df["job_title_norm"] = df["job_title"].map(norm_title)

    def _q25(s): return s.quantile(0.25)
    def _q75(s): return s.quantile(0.75)
    job_stats = (
        df.groupby(["job_title"], dropna=True)["salary"]
          .agg(salary_count="size",
               salary_mean="mean",
               salary_p25=_q25,
               salary_p75=_q75,
               salary_min="min",
               salary_max="max")
          .reset_index()
    )
    jobedu_stats = (
        df.groupby(["job_title","education_level"], dropna=True)["salary"]
          .agg(salary_count="size",
               salary_mean="mean",
               salary_p25=_q25,
               salary_p75=_q75,
               salary_min="min",
               salary_max="max")
          .reset_index()
    )
    for d in (job_stats, jobedu_stats):
        for c in ["salary_mean","salary_p25","salary_p75","salary_min","salary_max"]:
            d[c] = d[c].round(2)

    df.to_csv(CLEAN_CSV_PATH, index=False)
    job_stats.to_csv(JOB_STATS_PATH, index=False)
    jobedu_stats.to_csv(JOBEDU_STATS_PATH, index=False)
    log.info("✅ Kaggle cleaning & stats ready.")


# --------------------------
# Muse (probe for the role)
# --------------------------
def html_to_text(html: str) -> str:
    if not isinstance(html, str):
        return ""
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(" ", strip=True)
    return " ".join(text.split())

def is_remote_location(name: str) -> bool:
    return isinstance(name, str) and ("remote" in name.strip().lower() or "flexible" in name.strip().lower())

def is_us_location(name: str | None) -> bool:
    if not name:
        return False
    s = name.strip().lower()
    if "remote" in s and ("usa" in s or "united states" in s):
        return True
    if "united states" in s or "usa" in s or s.endswith(", us"):
        return True
    parts = [p.strip() for p in name.split(",")]
    if len(parts) >= 2 and parts[-1].strip().upper() in US_STATES:
        return True
    return False

def pick_display_location(locs: list[dict]) -> str:
    names = [(l or {}).get("name", "") for l in (locs or [])]
    for n in names:
        if is_remote_location(n):
            return n
    for n in names:
        if is_us_location(n):
            return n
    return names[0] if names else ""

def score_job_match(title: str, job: dict) -> float:
    tl = (title or "").lower()
    name = (job.get("name") or "").lower()
    if not tl or not name:
        return 0.0
    score = difflib.SequenceMatcher(None, tl, name).ratio()
    if tl in name:
        score += 0.08
    return score

def muse_search_jobs(pages: int = 6,
                     category: str | None = None,
                     location: str | None = None) -> list[dict]:
    results: list[dict] = []
    for page in range(1, pages + 1):
        params = {"page": page}
        if category: params["category"] = category
        if location: params["location"] = location
        try:
            data = _get_json(MUSE_BASE, params=params)
            batch = data.get("results", [])
            if not batch:
                break
            results.extend(batch)
        except Exception as e:
            log.warning(f"Muse page {page} failed: {e}")
    return results

SKILL_PATTERNS = [
    r"\bpython\b", r"\bsql\b", r"\bexcel\b", r"\btableau\b", r"\bpower\s*bi\b",
    r"\baws\b", r"\bazure\b", r"\bgcp\b", r"\bsnowflake\b",
    r"\bscala\b", r"\br\b", r"\bspark\b", r"\bhadoop\b",
    r"\bairflow\b", r"\bdocker\b", r"\bkubernetes\b",
    r"\bapi\b", r"\brest\b", r"\bgraphql\b",
    r"\bml\b", r"\bmachine learning\b", r"\bstatistics?\b",
    r"\bcommunication\b", r"\bpresentation\b", r"\bstakeholder\b", r"\bproject management\b",
]

def _pattern_to_label(pat: str) -> str:
    label = pat
    label = re.sub(r"\\\\b", "", label)
    label = re.sub(r"\\\\s\*", " ", label)
    label = re.sub(r"[()\\\\]", "", label)
    label = re.sub(r"\s+", " ", label).strip()
    if label.lower() == "power bi":
        return "Power BI"
    return label.upper() if len(label) <= 4 else label.title()

def extract_skills(texts: List[str], top_k: int = 12) -> List[str]:
    counts: Dict[str, int] = {}
    for t in texts:
        t = (t or "").lower()
        for pat in SKILL_PATTERNS:
            if re.search(pat, t):
                key = _pattern_to_label(pat)
                counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    return [k for k,_ in ranked[:top_k]]

def muse_probe_for_title(title: str,
                         pages: int = 6,
                         locations: Optional[List[str]] = None,
                         us_only: bool = True,
                         top_k_posts: int = 5) -> Dict:
    """Probe Muse for a single role and extract frequent skills."""
    title = normalize_text(title)
    locations = locations or ["Remote", "United States"]
    collected: List[dict] = []
    for loc in locations:
        collected += muse_search_jobs(pages=pages, category=None, location=loc)

    if us_only and collected:
        filtered = []
        for j in collected:
            locs = j.get("locations") or []
            if locs and all(
                is_us_location((l or {}).get("name", "")) or is_remote_location((l or {}).get("name", ""))
                for l in locs
            ):
                filtered.append(j)
        collected = filtered

    scored = sorted([(score_job_match(title, j), j) for j in collected],
                    key=lambda x: x[0], reverse=True)
    picked = [j for s, j in scored if s >= 0.50][:top_k_posts]

    if not picked:
        name_kw_patterns = [
            r"\bdata\s*scientist\b",
            r"\bmachine\s*learning\b",
            r"\bml\b",
            r"\bai\b",
            r"\bresearch\s*scientist\b",
            r"\b(applied|staff|senior|lead)\s+data\s*scientist\b",
        ]
        for _, j in scored:
            name = (j.get("name") or "").lower()
            if any(re.search(p, name) for p in name_kw_patterns):
                picked.append(j)
                if len(picked) >= top_k_posts:
                    break

    if not picked:
        picked = [j for _, j in scored[:top_k_posts]]

    texts, posts = [], []
    for j in picked:
        contents_txt = html_to_text(j.get("contents", ""))
        texts.append(contents_txt)
        posts.append({
            "muse_name": j.get("name",""),
            "company": (j.get("company") or {}).get("name",""),
            "location": pick_display_location(j.get("locations") or []),
            "level": (j.get("levels") or [{}])[0].get("name","") if j.get("levels") else "",
            "landing_page": (j.get("refs") or {}).get("landing_page",""),
        })

    skills = extract_skills(texts, top_k=12)
    jd_corpus = "\n\n---\n\n".join(texts)
    return {
        "posts": posts,
        "skills": skills,
        "picked_count": len(picked),
        "collected_count": len(collected),
        "jd_corpus": jd_corpus,
    }


# --------------------------
# Markdown report builders (Baseline & DeepSeek)
# --------------------------
def build_common_header(user, row_title, row_titleedu) -> List[str]:
    lines = [f"# Career Insights for **{user['job_title']}**"]
    lines.append("")
    lines.append(f"- **Education**: {user['education_level']}")
    lines.append(f"- **Experience**: {user['years_of_experience']} years")
    lines.append(f"- **Current Salary**: {fmt_money(user['salary'])}")
    lines.append("")
    lines.append("## Market Benchmarks")
    if row_title is not None:
        m = row_title
        dv = safe_div(user["salary"] - m["salary_mean"], m["salary_mean"])
        lines.append(f"- **Title mean**: {fmt_money(m['salary_mean'])} "
                     f"(P25={fmt_money(m['salary_p25'])}, P75={fmt_money(m['salary_p75'])})")
        if pd.notna(dv):
            cmp_word = "above" if dv > 0 else "below"
            lines.append(f"- **Position vs title mean**: {percent(abs(dv))} {cmp_word}")
    else:
        lines.append("- No exact match found for the title benchmark.")
    if row_titleedu is not None:
        m = row_titleedu
        dv = safe_div(user["salary"] - m["salary_mean"], m["salary_mean"])
        lines.append(f"- **Title+Education mean**: {fmt_money(m['salary_mean'])} "
                     f"(P25={fmt_money(m['salary_p25'])}, P75={fmt_money(m['salary_p75'])})")
        if pd.notna(dv):
            cmp_word = "above" if dv > 0 else "below"
            lines.append(f"- **Position vs title+education mean**: {percent(abs(dv))} {cmp_word}")
    lines.append("")
    return lines

def build_baseline_report(user, row_title, row_titleedu, probe: dict) -> str:
    lines = build_common_header(user, row_title, row_titleedu)

    # Section: Common skills (regex from JD)
    lines.append("## Common Skills from Recent JDs (baseline)")
    skills = (probe or {}).get("skills") or []
    if skills:
        for s in skills:
            disp = s.upper() if len(s) <= 4 else s.title()
            lines.append(f"- Consider strengthening: **{disp}** (if not already mastered)")
    else:
        lines.append("- Not enough JD samples or no clear skills extracted.")
    lines.append("")

    # Section: Potential uplift (simple heuristic)
    lines.append("## Potential Uplift (heuristic)")
    if row_title and row_titleedu:
        uplift = row_titleedu["salary_mean"] - row_title["salary_mean"]
        if pd.notna(uplift) and uplift != 0:
            dir_word = "higher" if uplift > 0 else "lower"
            lines.append(f"- In the sample, the title+education mean is **{fmt_money(abs(uplift))} {dir_word}** "
                         f"than the title mean.")
    if skills:
        lines.append("- Closing the JD high-frequency skills above is often positively correlated with higher pay.")
    lines.append("")

    # Section: Action plan (generic)
    lines.append("## Action Plan (generic)")
    lines.append("- Track 10–20 target JDs weekly; list concrete skill gaps.")
    lines.append("- Build 1–2 micro projects (1–2 weeks each) that prove skills; publish on GitHub/blog.")
    lines.append("- Seek responsibilities aligned with target JDs to gather measurable achievements.")
    lines.append("- Re-calibrate salary targets every 2 weeks; refresh company list & referrals.")
    lines.append("")

    return "\n".join(lines).strip() + "\n"

def build_deepseek_report(user, row_title, row_titleedu, enriched: dict) -> str:
    lines = build_common_header(user, row_title, row_titleedu)

    # Section: DeepSeek guidance
    lines.append("## DeepSeek Guidance (model-based)")
    lines.append(f"- **Positioning**: {enriched.get('pay_label', 'N/A')}")
    gaps = enriched.get("top_gaps") or []
    if isinstance(gaps, str): gaps = [gaps]
    if gaps:
        lines.append(f"- **Skills to strengthen/develop**: {', '.join(map(str, gaps))}")
    if enriched.get("plan"):
        lines.append(f"- **Plan**: {enriched['plan']}")
    if enriched.get("counterfactual_salary") is not None:
        lines.append(f"- **Counterfactual salary**: {fmt_money(float(enriched['counterfactual_salary']))}")
    lines.append("")

    # Section: Skills by toolkit (from DeepSeek JD extraction)
    sbt = enriched.get("skills_by_toolkit") or {}
    lines.append("## Skills by Toolkit (from JD text)")
    if sbt:
        for bucket, items in sbt.items():
            if items:
                lines.append(f"- **{bucket}**: {', '.join(items)}")
    else:
        lines.append("- No skills extracted from JD.")
    lines.append("")

    return "\n".join(lines).strip() + "\n"


# --------------------------
# Interactive flow
# --------------------------
def interactive_flow(args):
    print("Welcome to Career Insights (interactive) 👋\nPlease follow the prompts:\n")

    age = ask("Age (years)", validate=v_int_range(14, 80))
    edu = ask("Education (e.g., Bachelor's / Master's / PhD / High School)", validate=v_nonempty())
    edu = standardize_education(edu)
    job = ask("Job Title (e.g., Data Analyst / Software Engineer)", validate=v_nonempty())
    years = ask("Years of Experience (can be decimal)", validate=v_float_range(0, 60))
    salary = ask("Current Annual Salary (USD, commas allowed)", validate=v_money_pos())

    user = {
        "age": age,
        "education_level": edu,
        "job_title": normalize_text(job),
        "job_title_norm": norm_title(job),
        "years_of_experience": years,
        "salary": salary,
    }

    ensure_clean_data(force=args.force_rebuild)

    job_stats = pd.read_csv(JOB_STATS_PATH)
    jobedu_stats = pd.read_csv(JOBEDU_STATS_PATH)

    row_title = job_stats.loc[job_stats["job_title"].str.lower() == user["job_title"].lower()]
    if row_title.empty:
        cand = difflib.get_close_matches(user["job_title"], job_stats["job_title"].tolist(), n=1, cutoff=0.6)
        row_title = job_stats.loc[job_stats["job_title"] == cand[0]] if cand else pd.DataFrame()
    row_title = (row_title.iloc[0].to_dict() if not row_title.empty else None)

    row_titleedu = jobedu_stats.loc[
        (jobedu_stats["job_title"].str.lower() == user["job_title"].lower()) &
        (jobedu_stats["education_level"].astype(str).str.lower() == user["education_level"].lower())
    ]
    row_titleedu = (row_titleedu.iloc[0].to_dict() if not row_titleedu.empty else None)

    # Muse probe (can be disabled)
    muse_skills: list[str] = []
    probe: dict = {}
    if not args.no_muse:
        try:
            probe = muse_probe_for_title(
                title=user["job_title"],
                pages=args.muse_pages,
                locations=args.muse_locations or None,
                us_only=not args.no_us_only,
                top_k_posts=args.muse_top_posts,
            )
            muse_skills = probe.get("skills", []) or []
            print("\n🔎 Muse probe status")
            print("────────────────────")
            print(f"  · Collected posts: {probe.get('collected_count', 0)}")
            print(f"  · Picked posts:    {probe.get('picked_count', 0)}")
            print(f"  · Extracted skills: {len(muse_skills)}")
            top_posts = (probe.get("posts") or [])[:3]
            if top_posts:
                print("  · Example posts:")
                for j in top_posts:
                    print(f"    - {j.get('company','?')} | {j.get('muse_name','?')} | {j.get('location','?')}")
            else:
                print("  · No matching posts selected (title/location mismatch or JD incomplete).")
        except Exception as e:
            log.warning(f"Muse probe failed (skipping skills extraction): {e}")
            print("\n🔎 Muse probe status: FAILED (skipped skills extraction)")

    # -------- Baseline (no LLM) --------
    baseline_md = build_baseline_report(user, row_title, row_titleedu, probe)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", user["job_title"].lower()).strip("_") or "role"
    baseline_path = REPORTS_DIR / f"employee_baseline_{slug}.md"
    baseline_path.write_text(baseline_md, encoding="utf-8")

    print("\n✅ Baseline report saved:", baseline_path)

    # -------- DeepSeek (LLM) --------
    enriched: dict = {}
    if dse is None and not args.no_deepseek_api:
        raise RuntimeError("deepseek_enrichment import failed; cannot call DeepSeek.")
    if dse is not None:
        jd_skills = ", ".join(muse_skills) if muse_skills else ""
        jd_corpus = probe.get("jd_corpus", "") if isinstance(probe, dict) else ""

        row_for_llm = {
            "job_title": user["job_title"],
            "education_level": user["education_level"],
            "years_of_experience": user["years_of_experience"],
            "salary": user["salary"],
            "salary_mean_title": (row_title or {}).get("salary_mean"),
            "salary_mean_titleedu": (row_titleedu or {}).get("salary_mean"),
            "delta_vs_title_mean": safe_div(user["salary"] - (row_title or {}).get("salary_mean", np.nan),
                                            (row_title or {}).get("salary_mean", np.nan)),
            "delta_vs_titleedu_mean": safe_div(user["salary"] - (row_titleedu or {}).get("salary_mean", np.nan),
                                               (row_titleedu or {}).get("salary_mean", np.nan)),
            "jd_skills": jd_skills,
            "jd_corpus": jd_corpus,
        }

        enriched_list = dse.enrich_rows(
            rows=[pd.Series(row_for_llm)],
            model=args.deepseek_model,
            use_api=not args.no_deepseek_api,
        )
        enriched = enriched_list[0] if enriched_list else {}

        deepseek_md = build_deepseek_report(user, row_title, row_titleedu, enriched)
        deepseek_path = REPORTS_DIR / f"employee_deepseek_{slug}.md"
        deepseek_path.write_text(deepseek_md, encoding="utf-8")
        print("✅ DeepSeek report saved:", deepseek_path)

        # Persist wide row (enriched)
        ENRICHED_DIR.mkdir(parents=True, exist_ok=True)
        row_out = {
            "age": user["age"],
            "education_level": user["education_level"],
            "job_title": user["job_title"],
            "years_of_experience": user["years_of_experience"],
            "salary": user["salary"],
            "salary_mean_title":      (row_title or {}).get("salary_mean"),
            "salary_mean_titleedu":   (row_titleedu or {}).get("salary_mean"),
            "delta_vs_title_mean":    row_for_llm["delta_vs_title_mean"],
            "delta_vs_titleedu_mean": row_for_llm["delta_vs_titleedu_mean"],
            "pay_label": enriched.get("pay_label"),
            "top_gaps": ", ".join(enriched.get("top_gaps") or []),
            "plan": enriched.get("plan"),
            "counterfactual_salary": enriched.get("counterfactual_salary"),
            "jd_skills": row_for_llm.get("jd_skills", ""),
            "skills_by_toolkit": json.dumps(enriched.get("skills_by_toolkit") or {}, ensure_ascii=False),
        }
        df_one = pd.DataFrame([row_out])
        if ENRICHED_OUT_PATH.exists():
            df_one.to_csv(ENRICHED_OUT_PATH, mode="a", header=False, index=False)
        else:
            df_one.to_csv(ENRICHED_OUT_PATH, index=False)

        # Persist bucketed skills long table
        bucket_rows = []
        for bucket, skills in (enriched.get("skills_by_toolkit") or {}).items():
            for sk in (skills or []):
                bucket_rows.append({
                    "job_title": user["job_title"],
                    "education_level": user["education_level"],
                    "years_of_experience": user["years_of_experience"],
                    "toolkit": bucket,
                    "skill": sk,
                    "source": "deepseek_jd",
                })
        if bucket_rows:
            df_b = pd.DataFrame(bucket_rows)
            if BUCKET_OUT_PATH.exists():
                df_b.to_csv(BUCKET_OUT_PATH, mode="a", header=False, index=False)
            else:
                df_b.to_csv(BUCKET_OUT_PATH, index=False)

    print("\n🎉 Done.")


# --------------------------
# CLI
# --------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Interactive Career Insights (Baseline vs DeepSeek)")
    p.add_argument("--force-rebuild", action="store_true", help="Rebuild cleaned data from Kaggle")

    # Muse controls
    p.add_argument("--no-muse", action="store_true", help="Disable Muse probing")
    p.add_argument("--muse-pages", type=int, default=6, help="Pages per location to fetch from Muse")
    p.add_argument("--muse-top-posts", type=int, default=5, help="Top posts to keep after scoring")
    p.add_argument("--muse-locations", nargs="*", default=["Remote", "United States"], help="Muse locations")
    p.add_argument("--no-us-only", action="store_true", help="Disable US/Remote filtering (not recommended)")

    # DeepSeek switches
    p.add_argument("--deepseek-model", default="deepseek-chat", help="DeepSeek model name")
    p.add_argument("--no-deepseek-api", action="store_true", help="Disable DeepSeek API (only baseline)")

    return p.parse_args()


def main():
    args = parse_args()
    interactive_flow(args)


if __name__ == "__main__":
    main()
