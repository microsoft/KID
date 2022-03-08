SHELL=/bin/bash
PROJECT_NAME=KID
PROJECT_PATH=${PROJECT_NAME}/
LINT_PATHS=${PROJECT_PATH} test/ scripts/ setup.py

check_install = python3 -c "import $(1)" || pip3 install $(1) --upgrade
check_install_extra = python3 -c "import $(1)" || pip3 install $(2) --upgrade

pytest:
	$(call check_install, pytest)
	$(call check_install, pytest_cov)
	$(call check_install, pytest_xdist)
	pytest test --cov ${PROJECT_PATH} --durations 0 -v --cov-report term-missing

lint:
	$(call check_install, flake8)
	$(call check_install_extra, bugbear, flake8_bugbear)
	flake8 ${LINT_PATHS} --count --show-source --statistics

format:
	# sort imports
	$(call check_install, isort)
	isort ${LINT_PATHS}
	# reformat using yapf
	$(call check_install, yapf)
	yapf -ir ${LINT_PATHS}

check-codestyle:
	$(call check_install, isort)
	$(call check_install, yapf)
	isort --check ${LINT_PATHS} && yapf -r -d ${LINT_PATHS}

commit-checks: format lint mypy

.PHONY: clean mypy lint format check-codestyle commit-checks
