SHELL=/bin/bash -O extglob

.PHONY: init develop test docs

init:
	pip install .
	pip install -r requirements.txt

develop:
	python setup.py develop

test: init develop
	python setup.py check
	make docs
	flake8 docs test video699
	python setup.py test

docs:
	rm -f docs/!(index).rst
	rm -r -f docs/_build/html
	sphinx-apidoc -o docs video699
	make -C docs html
