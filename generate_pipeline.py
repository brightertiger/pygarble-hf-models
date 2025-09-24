#!/usr/bin/env python3
"""
Simple generation pipeline for creating training data.
"""

import os
import logging
from pathlib import Path

from src.generator.generate import main as generate_main
from src.generator.json_to_csv import main as csv_main

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('generation_pipeline.log')
        ]
    )
    return logging.getLogger(__name__)

def generate_data(scenario: str = "all", num_samples: int = 1000):
    """Generate training data using the generator scripts."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting data generation for scenario: {scenario}")
    
    # Available scenarios
    scenarios = {
        "normal": "prompt_normal_parsing.yaml",
        "gibberish": "prompt_gibberish.yaml", 
        "domain": "prompt_domain_specific.yaml"
    }
    
    if scenario == "all":
        # Generate data for all scenarios
        for scenario_name, prompt_file in scenarios.items():
            logger.info(f"Generating data for {scenario_name} scenario")
            
            try:
                # Direct function call - no sys.argv manipulation needed
                generate_main(
                    prompt_file=f'src/generator/prompts/{prompt_file}',
                    num_batches=num_samples // 10,  # Convert samples to batches
                    summary=True
                )
                logger.info(f"Successfully generated {scenario_name} data")
            except Exception as e:
                logger.error(f"Failed to generate {scenario_name} data: {e}")
                raise
    else:
        # Generate data for specific scenario
        if scenario not in scenarios:
            logger.error(f"Unknown scenario: {scenario}. Available: {list(scenarios.keys())}")
            return False
        
        prompt_file = scenarios[scenario]
        logger.info(f"Generating data for {scenario} scenario")
        
        try:
            generate_main(
                prompt_file=f'src/generator/prompts/{prompt_file}',
                num_batches=num_samples // 10,  # Convert samples to batches
                summary=True
            )
            logger.info(f"Successfully generated {scenario} data")
        except Exception as e:
            logger.error(f"Failed to generate {scenario} data: {e}")
            raise
    
    return True

def convert_to_csv():
    """Convert generated JSON data to CSV format for training."""
    logger = logging.getLogger(__name__)
    
    logger.info("Converting JSON data to CSV format")
    
    try:
        csv_main()
        logger.info("Successfully converted JSON to CSV")
    except Exception as e:
        logger.error(f"Failed to convert JSON to CSV: {e}")
        raise

def main():
    """Main generation pipeline."""
    logger = setup_logging()
    
    logger.info("Starting data generation pipeline")
    
    # Configuration
    scenario = os.getenv('GENERATION_SCENARIO', 'all')  # all, normal, gibberish, domain
    num_samples = int(os.getenv('NUM_SAMPLES', '1000'))
    
    try:
        # Step 1: Generate data
        logger.info(f"Generating {num_samples} samples for scenario: {scenario}")
        generate_data(scenario, num_samples)
        
        # Step 2: Convert to CSV
        logger.info("Converting generated data to CSV format")
        convert_to_csv()
        
        logger.info("Data generation pipeline completed successfully")
        logger.info("Generated files:")
        logger.info("  - data/train.csv")
        logger.info("  - data/validation.csv")
        
        logger.info("\n🚀 Next steps:")
        logger.info("  1. Review the generated data in data/train.csv")
        logger.info("  2. Run training pipeline: python train_pipeline.py")
        logger.info("  3. Adjust generation parameters if needed")
        
    except Exception as e:
        logger.error(f"Generation pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
