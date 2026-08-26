setup:
	python -m venv .venv

install:
	pip install -r requirements.txt

run:
	python run_pipeline.py

app:
	streamlit run app.py
