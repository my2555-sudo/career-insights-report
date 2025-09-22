# career-insights-report
######## Career Insights Report ##########

An interactive ETL pipeline that combines two distinct data sources:
1. Kaggle salary dataset (CSV)
2. The Muse job postings API (web API + scraping of job descriptions)

The pipeline cleans and aggregates salary data, scrapes real job descriptions, extracts skill requirements, and finally integrates DeepSeek AI enrichment to produce actionable career insights.

######## Features ########
- 1) Data Sourcing & Extraction**  
  - Robustly pulls from Kaggle CSV + Muse API.  
  - Raw data saved under `data/raw/`.  
- 2) ETL & Cleaning**  
  - `main.py` orchestrates the ETL workflow.  
  - Cleansing, normalization, and statistical aggregation done with Pandas.  
- 3) AI Enrichment**  
  - `deepseek_enrichment.py` encapsulates all AI-related logic.  
  - DeepSeek API parses job descriptions, infers missing skill gaps, and outputs structured recommendations.  
- 4) Reports**  
  - Results are exported as before/after Markdown reports in `examples/`.  
  - Side-by-side comparison between baseline vs. DeepSeek-enriched insights.  

######## Usage ########
```bash
# install dependencies
pip install -r requirements.txt
# run interactive CLI
python main.py


######## Project Structure ########
project-name/
├── main.py                 # orchestration script
├── deepseek_enrichment.py  # AI enrichment module
├── data/
│   ├── raw/                # original data
│   └── enriched/           # cleaned/enriched data
├── examples/               # before/after Markdown reports
├── README.md
├── DEEPSEEK_USAGE.md
├── AI_USAGE.md
├── requirements.txt
└── .gitignore
