.PHONY: install-dev install lint format help jupyter

help:
	@echo "Available tasks:"
	@echo "  make install-dev  -- Install development dependencies."
	@echo "  make install      -- Install production dependencies."
	@echo "  make lint         -- Run code linters."
	@echo "  make format       -- Run code formatters."
	@echo "  make jupyter      -- Start up JupyterLab."
	@echo "  make help         -- Display this help message."


install-dev:
	pip3 install pipenv --upgrade
	pipenv install --dev

install:
	pip3 install pipenv --upgrade
	pipenv install

lint:
	pipenv run flake8 notebooks
	pipenv run pylint --rcfile=.pylintrc notebooks
	pipenv run nbqa flake8 notebooks
	pipenv run nbqa pylint --rcfile=.pylintrc notebooks

format:
	pipenv run black notebooks
	pipenv run isort --atomic notebooks
	pipenv run nbqa black notebooks
	pipenv run nbqa isort --atomic notebooks

jupyter:
	pipenv run jupyter lab notebooks