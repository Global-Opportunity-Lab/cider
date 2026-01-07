# Makes a fresh install of the repository environment
fresh-env-pkg-dependencies:
	uv sync --no-dev --no-install-project

fresh-env-only-dev-dependencies:
	uv sync --only-dev

fresh-env-project-only:
	uv pip install -e .

fresh-env:
	@make fresh-env-pkg-dependencies
	@make fresh-env-only-dev-dependencies
	@make fresh-env-project-only
	pre-commit install

# Runs all tests
test:
	export JAVA_HOME="$(/usr/libexec/java_home -v 17)"
	export PATH="$JAVA_HOME/bin:$PATH"
	uv run pytest -v tests/


# Clears results from jupyter notebooks; results should not be commited as they contain binary blobs which bloat/obscure git history
clear-nb:
	@if [ "$(DIRS)" = "" ]; then \
		find . -not -path '*/\.*' -type f -name "*.ipynb" -exec uv run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} +; \
	else \
		find $(DIRS) -not -path '*/\.*' -type f -name "*.ipynb" -exec uv run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} +; \
	fi

# Delete Python cache files
clear-pycache:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
