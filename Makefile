.PHONY: all prepare train explain test api docker-build docker-run docker-compose-up docker-compose-down

all:
	python scripts/build_all.py

prepare:
	python scripts/build_all.py --prepare

train:
	python scripts/build_all.py --train

explain:
	python scripts/build_all.py --explain

test:
	pytest

api:
	uvicorn api.main:app --reload --port 8000

docker-build:
	docker build -t email-spam-api .

docker-run:
	docker run -p 8000:8000 email-spam-api

docker-compose-up:
	docker compose up --build

docker-compose-down:
	docker compose down
