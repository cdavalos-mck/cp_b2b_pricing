.PHONY: install test
SHELL=/bin/bash
NB_ENV=bm-sop-nb
LAST_VERSION=$$(ls dist/*.whl | sort -V | tail -n 1)
UV_VENV=. $(NB_ENV)/bin/activate

## build and install all components and requirements
install:
	uv sync --extra dev && \
	. .venv/bin/activate && \
	pre-commit install
	@echo "Environment .venv successfully created"

test:
	python -m pytest

nb-env:
	uv build
	uv venv $(NB_ENV)
	$(UV_VENV) && uv pip install $(LAST_VERSION)[dev]

docker-build:
	docker-compose --profile build up --build
	docker-compose --profile local build --parallel
	# Remove unused cache
	docker builder prune -f --filter=until=24h
	# Remove unused images
	docker image prune -f


docker-upload:
	docker-compose stop
	docker-compose --profile local up

docker-upload-silent:
	docker-compose stop
	docker-compose up -d

docker-run: docker-build docker-upload

docker-run-silent: docker-build docker-upload-silent

update-backend:
	docker-compose up package-build --build
	docker exec backend bash -c "source bm-sop/bin/activate && uv pip install $$(ls dist/*.whl | sort -V | tail -n 1) --force-reinstall"
	docker-compose restart backend
