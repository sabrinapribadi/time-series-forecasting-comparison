# Time Series Forecasting Model Comparison Toolkit

A comprehensive, reproducible benchmark for evaluating and comparing time series forecasting models.

## Project Structure
- `data/`: Raw, processed, and forecast data
- `notebooks/`: Jupyter notebooks for EDA and experimentation
- `src/`: Modular source code
- `configs/`: Configuration files
- `models/`: Saved model artifacts
- `dashboard/`: Interactive Streamlit dashboard
- `tests/`: Unit tests
- `docs/`: Documentation (PRD, ADR)

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Download sample data
python scripts/download_data.py

# Run full pipeline
python src/train.py
