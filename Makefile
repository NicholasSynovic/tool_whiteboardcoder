build:
	poetry build
	pip install dist/*.tar.gz

build-docs:
	sphinx-build --builder html src-docs build-docs

create-dev:
	pre-commit install
	rm -rf env
	python3.10 -m venv env
	( \
		. env/bin/activate; \
		pip install -r requirements.txt; \
		pip install torch torchvision torchaudio \
			--index-url https://download.pytorch.org/whl/cu124; \
		pip install flash-attn; \
		poetry install; \
		deactivate; \
	)

create-docs:
	sphinx-apidoc src --output-dir src-docs --maxdepth 100 --separate
