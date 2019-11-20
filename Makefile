test:
	py.test --pyargs AFQ --cov-report term-missing --cov=AFQ

devtest:
	py.test -x --pyargs AFQ --cov-report term-missing --cov=AFQ

pdb:
	py.test --pyargs AFQ --cov-report term-missing --cov=AFQ --pdb

flake8:
	flake8 --ignore N802,N806,W503 --select W504 `find . -name \*.py | grep -v setup.py | grep -v version.py | grep -v __init__.py | grep -v /docs/`

all: test flake8