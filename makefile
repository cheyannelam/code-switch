install:
	git config core.hooksPath .githooks
	- git submodule update --init --recursive --remote --force
	pip install -e .
	mkdir -p build && cd build && \
	  cmake ../kenlm && make -j

lint-format:
	git ls-files '*.py' | xargs -t autoflake --in-place --expand-star-imports --remove-all-unused-imports --ignore-init-module-imports
	git ls-files '*.py' | xargs -t isort -q
	git ls-files '*.py' | xargs -t black -q
	git ls-files '*.yml' '*.yaml' | xargs -t yq -i -S -Y -w 10000 .
	git ls-files '.gitignore' | LC_ALL="C" xargs -tI {} sort -o {} {}

lint:
	git ls-files '*.py' | xargs -t flake8
	git ls-files '*.py' | xargs -t pylint

reset:
	$(MAKE) install
