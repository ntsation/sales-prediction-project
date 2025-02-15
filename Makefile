VENV_DIR = .venv
REQ_FILE = requirements.txt
LOG_FILE = logs/log.txt
MODEL_FILE = models/sales_model.pkl
DATA_FILE = data/sales_data.csv

# Initializes the virtual environment and installs dependencies
init:
	@echo "🔧 Creating virtual environment..."
	python3 -m venv $(VENV_DIR)
	@echo "✅ Virtual environment created!"
	@echo "📦 Installing dependencies..."
	$(VENV_DIR)/bin/pip install --upgrade pip
	$(VENV_DIR)/bin/pip install -r $(REQ_FILE)
	@echo "✅ Dependencies installed!"

# Runs the model training
train: $(DATA_FILE)
	@echo "📊 Starting model training..."
	$(VENV_DIR)/bin/python train.py
	@echo "✅ Training completed!"

# Runs the FastAPI backend
backend:
	@echo "🚀 Starting FastAPI API..."
	$(VENV_DIR)/bin/uvicorn api.main:app --reload

# Cleans up generated files and directories
clean:
	@echo "🧹 Removing files and directories..."
	rm -rf $(VENV_DIR)
	rm -rf __pycache__
	rm -rf logs/*
	rm -rf models/*
	@echo "✅ Cleanup completed!"
