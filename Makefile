
install:
	python setup.py install --user

test: install
	pytest .

clean:
	rm -rf dist/ build/
