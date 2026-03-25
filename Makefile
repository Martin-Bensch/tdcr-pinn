PYTHON ?= python3

.PHONY: setup setup-binding test smoke

setup:
	bash scripts/bootstrap.sh

setup-binding:
	$(PYTHON) -m pip install -e ./Required/tdcr-lilge-binding

test:
	$(PYTHON) -m pytest -q

smoke:
	$(PYTHON) -c "import pytdcrpinn; import pytdcrsv; print('smoke-ok')"
