.PHONY: init develop test docs

init:
	pip install -r requirements.txt

develop:
	python setup.py develop

test: init develop
	make docs
	flake8 docs test video699
	python setup.py test

docs:
	sphinx-apidoc -o docs video699
	make -C docs html
