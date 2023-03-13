.PHONY: help prepare-dev prepare-prod test doc clean

clean:
	rm -rf .tox

python-dev:
	python3 -m venv env
	env/bin/python -m pip install -r requirements.txt
	touch .env
