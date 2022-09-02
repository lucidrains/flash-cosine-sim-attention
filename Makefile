
install:
	python setup.py install --user

test: install
	python test.py

clean:
	rm -rf dist/ build/
