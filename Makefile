test:
	py.test --pyargs AFQ --cov-report term-missing --cov=AFQ

flake8:
	flake8 --ignore N802,N806 `find . -name \*.py | grep -v setup.py | grep -v version.py | grep -v __init__.py | grep -v /doc/`

all: test flake8