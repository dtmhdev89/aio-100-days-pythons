# Variable
PYTHON := python3
VENV_NAME := .venv
VENV_BIN = $(VENV_NAME)/bin
PIP := pip3

VIRTUALENV_PACKAGE := virtualenv
VIRTUALENV_CMD := virtualenv

# install virtualenv
install_virtualenv:
	$(PIP) install $(VIRTUALENV_PACKAGE)

venv: install_virtualenv
	$(VIRTUALENV_CMD) $(VENV_NAME) -p $(PYTHON)

install_libs:
	$(VENV_BIN)/$(PIP) install -q torch==2.4 torchvision==0.19 numpy \
		matplotlib ipykernel nltk  pillow tokenizers

export_requirments:
	$(VENV_BIN)/$(PIP) freeze > requirements.txt

venv_cleanup:
	@if [ -d "$(VENV_NAME)" ]; then \
		rm -rf $(VENV_NAME); \
		echo "Removed $(VENV_NAME)"; \
	else \
		echo "$(VENV_NAME) not found, nothing to remove."; \
	fi

run_base_model: base_model.py
	$(VENV_BIN)/$(PYTHON) base_model.py

.PHONY: install_virtualenv venv install_libs export_requirments
