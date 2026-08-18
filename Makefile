.PHONY: install train-baseline train-classifier test run docker-build docker-run

install:
	pip install -r requirements.txt

train-baseline:
	python ml/train_baseline.py

train-classifier:
	python ml/train_classifier.py

evaluate:
	python ml/evaluate_classifier.py

test:
	pytest tests/ -v

run:
	python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t medreport-copilot .

docker-run:
	docker run --rm -p 8000:8000 --env-file .env medreport-copilot
