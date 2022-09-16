
install:
	python setup.py install --user

test: install
	python setup.py test

benchmark: install
	python benchmark.py

benchmark_forward: install
	python benchmark.py --only-forwards

benchmark_backward: install
	python benchmark.py --only-backwards

train: install
	python train.py --use-cuda-kernel

clean:
	rm -rf dist/ build/
