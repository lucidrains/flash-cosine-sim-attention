
install:
	python setup.py install --user

test: install
	python setup.py test

benchmark: install
	python benchmark.py

benchmark_causal: install
	python benchmark.py --causal

benchmark_forward: install
	python benchmark.py --only-forwards

benchmark_backward: install
	python benchmark.py --only-backwards

benchmark_forward_causal: install
	python benchmark.py --only-forwards --causal

benchmark_backward_causal: install
	python benchmark.py --only-backwards --causal

train: install
	python train.py --use-cuda-kernel

clean:
	rm -rf dist/ build/
