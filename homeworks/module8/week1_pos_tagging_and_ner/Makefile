# Variables
PYTHON := python3
VENV_NAME := .venv
VENV_BIN = $(VENV_NAME)/bin
PIP := pip3

VIRTUALENV_PACKAGE := virtualenv
VIRTUALENV_CMD := virtualenv
TARGET_ARGS :=
DATA_URL := "https://figshare.com/ndownloader/files/17493650"
DATA_OUTPUT_NAME := MACCROBAT2018.zip
DATA_FOLDER := MACCROBAT2018

# install system packages
install_system_packages:
ifeq ($(OS), Linux)
	@echo "Running on Linux"
	apt install -y curl unzip
endif

ifeq ($(OS), Darwin)
	@echo "Runing on macOS"
	brew install curl unzip
endif

ifeq ($(OS), Windows_NT)
	@echo "Running on Windows"
	@echo "Not supported at the moment. Manually install it please."
endif
	
# install virtualenv
install_virtualenv:
	$(PIP) install $(VIRTUALENV_PACKAGE)

venv: install_virtualenv
	$(VIRTUALENV_CMD) $(VENV_NAME) -p $(PYTHON)

install_libs:
	$(VENV_BIN)/$(PIP) install -q numpy matplotlib ipykernel \
		scikit-learn torch transformers evaluate accelerate nltk

install: requirements.txt
	$(VENV_BIN)/$(PIP) install -r requirements.txt

export_requirments:
	$(VENV_BIN)/$(PIP) freeze > requirements.txt

download_and_unzip_data:
	mkdir $(DATA_FOLDER)
	curl -L -o $(DATA_OUTPUT_NAME) $(DATA_URL)
	unzip $(DATA_OUTPUT_NAME) -d $(DATA_FOLDER)

data_cleanup:
	@if [ -d "$(DATA_FOLDER)" ]; then \
		rm -rf $(DATA_FOLDER); \
		echo "Removed $(DATA_FOLDER)"; \
	else \
		echo "$(DATA_FOLDER) not found, nothing to remove."; \
	fi

	@if [ -f "$(DATA_OUTPUT_NAME)" ]; then \
		rm "$(DATA_OUTPUT_NAME)"; \
		echo "Deleted $(DATA_OUTPUT_NAME)"; \
	fi

venv_cleanup:
	@if [ -d "$(VENV_NAME)" ]; then \
		rm -rf $(VENV_NAME); \
		echo "Removed $(VENV_NAME)"; \
	else \
		echo "$(VENV_NAME) not found, nothing to remove."; \
	fi

run_1_pos_tagging: 1_pos_tagging.py
	$(VENV_BIN)/$(PYTHON) 1_medical_ner.py

run_2_medical_ner: 2_medical_ner.py
	$(VENV_BIN)/$(PYTHON) 2_medical_ner.py

.PHONY: install_virtualenv venv install_libs export_requirments run_1_medical_ner
