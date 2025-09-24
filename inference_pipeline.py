#!/usr/bin/env python3
"""
Inference and evaluation pipeline script.
Loads configuration and runs comprehensive model evaluation.
"""

import yaml
import logging
from pathlib import Path
from src.inference.evaluation import ModelEvaluator

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('inference_evaluation.log')
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = "src/inference/config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    """Main evaluation function."""
    logger = setup_logging()
    
    # Load configuration
    config_path = "src/inference/config.yaml"
    config = load_config(config_path)
    
    logger.info("Starting detailed model evaluation...")
    logger.info(f"Output directory: {config['evaluation']['output_dir']}")
    
    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(config)
        
        # Run evaluation
        results = evaluator.run_evaluation()
        
        # Generate outputs
        evaluator.generate_visualizations(results)
        evaluator.generate_markdown_report(results)
        evaluator.save_detailed_results(results)
        
        output_dir = config['evaluation']['output_dir']
        logger.info(f"Evaluation completed successfully!")
        logger.info(f"Results saved to: {output_dir}/")
        logger.info(f"View the report: {output_dir}/evaluation_report.md")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
