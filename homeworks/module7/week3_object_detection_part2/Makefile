# Variable
PYTHON = python3
VENV_NAME = .venv
VENV_BIN = $(VENV_NAME)/bin
PIP = pip3
VIRTUALENV_PACKAGE := virtualenv
VIRTUALENV_CMD := virtualenv

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
