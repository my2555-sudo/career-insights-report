This file shows how DeepSeek AI was integrated into the pipeline.

## Prompts
The enrichment module sends structured prompts containing:
- Role, education, experience, salary
- Market salary benchmarks
- Extracted job description corpus (from The Muse API)
- Regex-mined skills as side-signals

### Prompt Goals
1. Generate a one-line pay positioning.  
2. Extract top 3 missing skills.  
3. Recommend a 2-year career development plan.  
4. Provide a counterfactual salary estimate if skills are closed.  
5. Categorize extracted JD skills into toolkits (`Programming`, `Cloud`, `Visualization`, etc.).

## Reasoning
- **Why DeepSeek?**: Manual rule-based heuristics cannot capture nuanced skills from raw JD text.  
- **Challenge**: JD text is noisy, inconsistent, and often lengthy.  
- **Solution**: Use DeepSeek to parse free text into **structured JSON** (`pay_label`, `top_gaps`, `plan`, `counterfactual_salary`, `skills_by_toolkit`).  

## Output Integration
- Enriched rows are merged back into `data/enriched/salary_enriched.csv`.  
- Skills-by-toolkit are appended to `data/enriched/muse_skills_bucketed.csv`.  
- Markdown comparison reports are written to `examples/`.  
