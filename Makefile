# Tests, figures and generated tables from the saved results.
PY ?= python3

.PHONY: test figures

test:
	$(PY) -m pytest -q tests

figures:
	$(PY) scripts/m4_diagram.py > /dev/null
	$(PY) scripts/m5_figures.py
	$(PY) scripts/m6_estimates.py --tables
