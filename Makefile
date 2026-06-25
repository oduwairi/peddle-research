.PHONY: lint typecheck test format all train-smoke train-inspect

all: lint typecheck test

lint:
	ruff check src/ tests/ scripts/

format:
	ruff check --fix src/ tests/ scripts/
	ruff format src/ tests/ scripts/

typecheck:
	mypy src/draper/

test:
	pytest tests/ -v

# QLoRA smoke checks — run these before paying for cloud GPU
train-inspect:
	uv run python scripts/train.py inspect

train-smoke:
	uv run python scripts/train.py smoke --dry-run

# Upload project files (src, scripts, configs, data/final) from this laptop
# to the rented GPU pod. Requires POD_HOST (and usually POD_PORT) in env.
train-upload:
	bash scripts/upload_to_pod.sh

# Cloud-pod bootstrap: run this ON the pod after upload, with HF_TOKEN +
# HF_HUB_REPO exported.
train-cloud:
	bash scripts/train_cloud.sh
