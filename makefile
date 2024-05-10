install:
	git config core.hooksPath .githooks
	pip install -e .

lint-format:
	export LC_ALL="C"
	git ls-files '*.py' | xargs -t autoflake --in-place --expand-star-imports --remove-all-unused-imports --ignore-init-module-imports
	git ls-files '*.py' | xargs -t isort -q
	git ls-files '*.py' | xargs -t black -q
	git ls-files '*.yml' '*.yaml' | xargs -t yq -i -S -Y -w 10000 .
	git ls-files '.gitignore' | xargs -tI {} sort -o {} {}

lint:
	git ls-files '*.py' | xargs -t flake8
	git ls-files '*.py' | xargs -t pylint

reset:
	$(MAKE) install
