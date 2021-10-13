# Runs all tests
test:
	poetry run ./check_for_unmarked_tests.sh
	poetry run pytest $(filter-out $@,$(MAKECMDGOALS))

# Run lint checks. This will sort/format any files that arn't already formatted
lint:
	poetry run isort --profile black $(filter-out $@,$(MAKECMDGOALS))
	poetry run black $(filter-out $@,$(MAKECMDGOALS))
	poetry run autoflake -i -r --remove-all-unused-imports --remove-duplicate-keys \
	                     --remove-unused-variables --ignore-init-module-imports $(filter-out $@,$(MAKECMDGOALS))
	poetry run mypy $(filter-out $@,$(MAKECMDGOALS))

# Runs lint checks. Instead of automatically formatting, this will fail if any files aren't linted
# correctly.
lint-dryrun:
	poetry run isort --profile black $(filter-out $@,$(MAKECMDGOALS)) -c
	poetry run black $(filter-out $@,$(MAKECMDGOALS)) --check
	poetry run autoflake -r --remove-all-unused-imports --remove-duplicate-keys \
	                     --remove-unused-variables --ignore-init-module-imports $(filter-out $@,$(MAKECMDGOALS))
	poetry run mypy $(filter-out $@,$(MAKECMDGOALS))

# Clears results from jupyter notebooks; results should not be commited as they contain binary blobs which bloat/obscure git history
clear-nb:
	find $(filter-out $@,$(MAKECMDGOALS)) -not -path '*/\.*' -type f -name "*.ipynb" -execdir jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} +

# Dummy command, needed to interpret multiple words as args rather than commands. See https://stackoverflow.com/questions/6273608/how-to-pass-argument-to-makefile-from-command-line
%:      
    @:    
