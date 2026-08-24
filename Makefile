
install:
	pip install -e .

install_requirements:
	pip install -r requirements.txt

test: install
	python -m pytest

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

train: install install_requirements
	python train.py --use-cuda-kernel

train_triton: install install_requirements
	python train.py --use-triton

clean:
	rm -rf dist/ build/ *.egg-info
