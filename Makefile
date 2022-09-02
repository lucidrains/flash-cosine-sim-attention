
install:
	python setup.py install --user

test: install
	python setup.py test

benchmark: install
	python benchmark.py

clean:
	rm -rf dist/ build/
