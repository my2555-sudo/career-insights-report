##### AI Usage in Development #####

1）Development Stages
- Scaffolding: AI tools were used to generate boilerplate code such as API calls and CSV handling.
- Iteration: Human refinement was necessary to ensure correctness and handle edge cases.
- Prompt Engineering: Prompts were revised multiple times to achieve consistent JSON output and accurate skill categorization.

2）What AI Could Not Handle
- Logical integration issues: When connecting two external APIs (The Muse and DeepSeek) with the local data pipeline, the logic produced by AI often contained errors or gaps.
- Over-complication: For relatively simple problems, AI tended to produce overly complex or redundant code, which made maintenance harder.
- ETL construction failure: AI was unable to set up a complete ETL pipeline (main.py), and the output it generated was often incomplete or inconsistent。
- Skipping critical steps: To “save effort,” AI sometimes skipped over important tasks like data cleaning, outlier handling, or error management, which made results unreliable.
- Lack of reproducibility: With the same prompt, the AI output was not always consistent, making it difficult to reproduce results across runs.

3）Reflections
- AI did speed up the coding process, but it also meant more time had to be spent debugging and fixing bugs.
- Human intervention was especially important in the following areas:
    1. Logical flow: Making sure the interactions between different APIs and the local data were correct.
    2. Result validation: Ensuring that CSV/Markdown outputs were complete and followed a consistent schema.
    3. Data cleaning: Adding proper steps such as outlier winsorization and regex extraction that AI often ignored.
    4. Robustness: Introducing error handling and fallback mechanisms so the script could still run under API failures or network issues.
    5. Maintainability: Simplifying and modularizing AI-generated code to improve readability and make future changes easier.