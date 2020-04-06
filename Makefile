SHELL=/bin/bash -O extglob

.PHONY: init test docs

init:
	pip install -U pip setuptools
	pip install -e .

test: init
	python setup.py check
	make docs
	flake8 docs test video699
	python setup.py test
	coverage run -m unittest discover -s test/
	coverage report -m
	codecov -t 0a5f01f0-3df8-412a-afd7-7ab9a45d1fdd

docs:
	pip install -r docs/requirements.txt
	rm -r -f docs/_build/html
	sphinx-apidoc -o docs video699
	make -C docs html
