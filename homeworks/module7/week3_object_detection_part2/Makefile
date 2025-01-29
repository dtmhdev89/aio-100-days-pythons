# Variable
PYTHON = python3
VENV_NAME = .venv
VENV_BIN = $(VENV_NAME)/bin
PIP = pip3
VIRTUALENV_PACKAGE := virtualenv
VIRTUALENV_CMD := virtualenv

TEMP_REPO := temp_repo
REMOTE_REPO :=
REMOTE_FOLDER_PATH :=

# make temp repo for accessing on other platform like colab or kaggle
load_code_on_temp_repo:
	@if [ -z "$(TEMP_REPO)" ] || [ -z "$(REMOTE_REPO)" ] || [ -z "$(REMOTE_FOLDER_PATH)" ]; then \
		echo "Error: One or more required variables are not set."; \
	else \
		!git init $(TEMP_REPO)
		!cd $(TEMP_REPO) && git remote add origin $(REMOTE_REPO)
		!cd $(TEMP_REPO) && git config core.sparseCheckout true
		!echo "$(REMOTE_FOLDER_PATH)/*" > $(TEMP_REPO)/.git/info/sparse-checkout
		!cd $(TEMP_REPO) && git pull
	fi

# install virtualenv
install_virtualenv:
	$(VENV_BIN)/$(PIP) install $(VIRTUALENV_PACKAGE)

venv: install_virtualenv
	$(VIRTUALENV_CMD) $(VENV_NAME) -p $(PYTHON)

install_libs:
	$(VENV_BIN)/$(PIP) install -q numpy opencv-python ultralytics \
		matplotlib ipykernel

install: requirements.txt
	$(VENV_BIN)/$(PIP) install -r requirements.txt

export_requirments:
	$(VENV_BIN)/$(PIP) freeze > requirements.txt

run_1_obj_tracking_simple: 1_object_tracking_simple_vesion.py

.PHONY: install_virtualenv venv install_libs export_requirments run_1_obj_tracking_simple
