# Sentence Transformer Classifier Makefile

.PHONY: help install setup train evaluate publish clean test demo

help: ## Show this help message
	@echo "Sentence Transformer Classifier"
	@echo "================================"
	@echo ""
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

setup: ## Run complete setup and test
	@echo "Setting up Sentence Transformer Classifier..."
	chmod +x train evaluate publish
	./train --create-sample-data
	./train
	./evaluate --model checkpoints/best-model-*.ckpt --test-file data/test.csv

train: ## Train model with default configuration
	./train

train-prod: ## Train model with production configuration
	./train --config src/configs/production.yaml

train-research: ## Train model with research configuration
	./train --config src/configs/research.yaml

evaluate: ## Evaluate model (requires checkpoint)
	@if [ ! -f checkpoints/best-model-*.ckpt ]; then \
		echo "No checkpoint found. Run 'make train' first."; \
		exit 1; \
	fi
	./evaluate --model checkpoints/best-model-*.ckpt --test-file data/test.csv

predict: ## Make prediction on single text (requires checkpoint)
	@if [ ! -f checkpoints/best-model-*.ckpt ]; then \
		echo "No checkpoint found. Run 'make train' first."; \
		exit 1; \
	fi
	@echo "Enter text to classify:"
	@read text; \
	./evaluate --model checkpoints/best-model-*.ckpt --text "$$text"

publish: ## Publish model to Hugging Face Hub (requires checkpoint)
	@if [ ! -f checkpoints/best-model-*.ckpt ]; then \
		echo "No checkpoint found. Run 'make train' first."; \
		exit 1; \
	fi
	@echo "Enter repository name (e.g., username/model-name):"
	@read repo; \
	./publish --checkpoint checkpoints/best-model-*.ckpt --repo-name $$repo

demo: ## Run complete demo pipeline
	@echo "Running demo pipeline..."
	./train --create-sample-data
	./train
	./evaluate --model checkpoints/best-model-*.ckpt --test-file data/test.csv
	@echo "Demo completed!"

test: ## Run tests (if available)
	@echo "Running basic functionality test..."
	./train --create-sample-data
	./train --config src/configs/default.yaml
	./evaluate --model checkpoints/best-model-*.ckpt --test-file data/test.csv
	@echo "✅ All tests passed!"

clean: ## Clean up generated files
	rm -rf checkpoints/
	rm -rf evaluation_results/
	rm -rf hf_model/
	rm -rf data/
	rm -rf logs/
	rm -rf lightning_logs/
	rm -rf wandb/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

clean-data: ## Clean up only data files
	rm -rf data/

clean-models: ## Clean up only model files
	rm -rf checkpoints/
	rm -rf hf_model/

clean-logs: ## Clean up only log files
	rm -rf logs/
	rm -rf lightning_logs/
	rm -rf wandb/

package: ## Create distribution package
	python setup.py sdist bdist_wheel

install-dev: ## Install package in development mode
	pip install -e .

uninstall: ## Uninstall package
	pip uninstall sentence-transformer-classifier -y
