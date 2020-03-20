SHELL=/bin/bash -O extglob

.PHONY: init test docs

init:
	pip install -U pip setuptools
	pip install -e .
	pip install -r requirements.txt

test: init
	python setup.py check
	make docs
	flake8 docs test video699
	python setup.py test

docs:
	rm -r -f docs/_build/html
	sphinx-apidoc -o docs video699
	make -C docs html
