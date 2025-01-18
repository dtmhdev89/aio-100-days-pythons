# Variables
OS := $(shell uname -s)
PYTHON_VERSION := $(shell python3 --version | awk '{print $$2}' | cut -d. -f1,2)
VENV_PACKAGE := python$(PYTHON_VERSION)-venv
PYTHON = python3
PIP = pip3
VENV_NAME := .venv
VENV_BIN = $(VENV_NAME)/bin
PYTHON_TARGET_VERSION = 3.12

# Create virtual environment
install_venv:
	@echo "Detected Python version: $(PYTHON_VERSION)"
	@echo "Install package: $(VENV_PACKAGE)"
	
ifeq ($(OS), Linux)
	@echo "Running on Linux"
	apt-get install -y $(VENV_PACKAGE)
endif

ifeq ($(OS), Darwin)
	@echo "Running on macOS"
	@echo "It's built in Python3. Using virtualenv if using older Python version"
endif

ifeq ($(OS), Windows_NT)
	@echo "Running on Windows"
	@echo "Not supported install venv in Windows at the moment. Manually install it please."
endif

venv: install_venv
	$(PYTHON) -m venv $(VENV_NAME)

install_libs:
	$(VENV_BIN)/$(PIP) install matplotlib ipykernel \
		numpy \
		torch torchsummary torchvision

export_libs:
	$(VENV_BIN)/$(PIP) freeze > requirements.txt

install: requirements.txt check_python_version
	$(VENV_BIN)/$(PIP) install -r requirements.txt

check_python_version:
	@if [ "$(PYTHON_VERSION)" != "$(PYTHON_TARGET_VERSION)" ]; then \
		echo "Error: Python $(PYTHON_TARGET_VERSION) is prefered, but $(PYTHON_VERSION) is installed."; \
	else \
		echo "Python version: $(PYTHON_VERSION)"; \
	fi

clean:
	@if [ -d "$(VENV_NAME)" ]; then \
		rm -rf $(VENV_NAME); \
		echo "Removed $(VENV_NAME)"; \
	else \
		echo "$(VENV_NAME) not found, nothing to remove."; \
	fi

run_test_1_cnn_segmentation: 1_CNN_segmentation.py
	$(VENV_BIN)/$(PYTHON) 1_CNN_segmentation.py

.PHONY: install_venv venv install_libs export_libs clean
