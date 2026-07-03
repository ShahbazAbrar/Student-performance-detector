# Separate Streamlit Frontend

This folder contains a small Streamlit app that reads the generated outputs from the main project.

## What it does
- Shows model comparison from `Output/model_results.csv`
- Shows prediction output from `Output/prediction_output.json`
- Shows search path from `Output/search_path.json`
- Shows CSP solution from `Output/csp_solution.json`
- Shows knowledge-based inference from `Output/knowledge_inference.json`
- Displays saved figures from `Output/Figures/`
- Shows the final text summary from `Output/final_system_summary.txt`

## Run
Create a separate virtual environment for this frontend and install dependencies from `Frontend/requirements.txt`.

Example on Windows PowerShell:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Set the output folder path in the sidebar to:
`D:/Projects/Student Performance AI ML Project/Output`

## Important
This frontend is separated from the core course code. It only reads files and does not retrain models or modify outputs.
