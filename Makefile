.PHONY: install clean data train test dashboard

install:
	pip install -r requirements.txt

data:
	python scripts/download_data.py

train:
	python src/train.py

test:
	pytest tests/

dashboard:
	streamlit run dashboard/app.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
