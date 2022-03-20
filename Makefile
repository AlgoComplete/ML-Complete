# Arcane incantation to print all the other targets, from https://stackoverflow.com/a/26339924
help:
	@$(MAKE) -pRrq -f $(lastword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/^# File/,/^# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | egrep -v -e '^[^[:alnum:]]' -e '^$@$$'

# Recipe for activating the conda environment within the sub-shell as a target, from https://stackoverflow.com/a/55696820/13749426
.ONESHELL:

# Need to specify bash in order for conda activate to work, otherwise it will try to use the default shell, which is "zsh" in this case
SHELL = /bin/zsh

# Note that the extra activate is needed to ensure that the activate floats env to the front of PATH, otherwise it will not work
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

# Create conda env from env.yml and compile and install exact pip packages
conda-pip:
	conda env update --prune -f env.yml
	$(CONDA_ACTIVATE) MLBasic
	pip-compile requirements/req.in
	pip-sync requirements/req.txt


# Linting python code with flake8
flake8-badge:
	flake8 ./ml_complete --exit-zero --format=html --htmldir ./reports/flake8 --statistics --tee --output-file ./reports/flake8/flake8stats.txt
	genbadge flake8 -o ./reports/flake8/flake8-badge.svg