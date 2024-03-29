SRC = arrayfire

ifeq ($(shell uname),Darwin)
ifeq ($(shell which gsed),)
$(error Please install GNU sed with 'brew install gnu-sed')
else
SED = gsed
endif
else
SED = sed
endif

.PHONY : version
version : 
	@python -c 'from arrayfire.version import VERSION; print(f"ArrayFire Python v{VERSION}")'

.PHONY : install
install :
	pip install --upgrade pip
	pip install pip-tools
	pip-compile requirements.in -o final_requirements.txt --allow-unsafe --rebuild --verbose
	pip install -e . -r final_requirements.txt

# Testing

.PHONY : flake8
flake8 :
	flake8 arrayfire tests examples

.PHONY : import-sort
import-sort :
	isort arrayfire tests examples

.PHONY : typecheck
typecheck :
	mypy arrayfire tests examples --cache-dir=/dev/null

.PHONY : tests
tests :
	pytest --color=yes -v -rf --durations=40 --cov-config=.coveragerc --cov=$(SRC) --cov-report=xml

# Cleaning

.PHONY : clean
clean :
	rm -rf .pytest_cache/
	rm -rf arrayfire.egg-info/
	rm -rf dist/
	rm -rf build/
	find . | grep -E '(\.mypy_cache|__pycache__|\.pyc|\.pyo$$)' | xargs rm -rf
