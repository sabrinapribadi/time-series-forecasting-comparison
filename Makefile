.PHONY: install data validate train-all train-multivariate train-tuned inference dashboard test clean

install:
	poetry install

data:
	poetry run python scripts/download_data.py

validate:
	poetry run python scripts/validate_data.py

# Train all models univariate (baseline benchmark, ETTh1)
train-all:
	for model in holt_winters arima prophet random_forest xgboost lightgbm catboost lstm transformer nbeats tft; do \
		poetry run python scripts/train_model.py --model $$model --variant h1; \
	done

# Train ML models in multivariate mode (HUFL/HULL/MUFL/MULL/LUFL/LULL as covariates)
train-multivariate:
	for model in random_forest xgboost lightgbm catboost; do \
		poetry run python scripts/train_model.py --model $$model --variant h1 --mode multivariate; \
	done

# Train tunable models with Optuna HPO (IOH AOP2026 trial counts)
train-tuned:
	poetry run python scripts/train_model.py --model holt_winters --variant h1 --tune
	poetry run python scripts/train_model.py --model random_forest --variant h1 --tune
	poetry run python scripts/train_model.py --model catboost --variant h1 --tune
	poetry run python scripts/train_model.py --model xgboost --variant h1 --tune

# Export inference results for dashboard
inference:
	poetry run python scripts/run_inference.py --all --variant h1

dashboard:
	PYTHONPATH=. poetry run streamlit run src/ui/dashboard.py

test:
	poetry run pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
