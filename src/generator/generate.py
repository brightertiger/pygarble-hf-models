#!/usr/bin/env python3

import os
import sys
import yaml
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any
import google.generativeai as genai
from json_repair import repair_json
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_prompt_config(prompt_file: str) -> Dict[str, Any]:
    """Load prompt configuration from YAML file."""
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error(f"Prompt file '{prompt_file}' not found.")
        sys.exit(1)
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file '{prompt_file}': {e}")
        sys.exit(1)

def setup_genai(api_key: str = None) -> None:
    """Setup Google GenAI with API key."""
    if api_key:
        genai.configure(api_key=api_key)
    else:
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            logger.error("Google API key not provided. Set GOOGLE_API_KEY environment variable or use --api-key")
            sys.exit(1)
        genai.configure(api_key=api_key)

def generate_content(prompt_config: Dict[str, Any], num_batches: int = 1, prompt_name: str = None) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Generate content using Google GenAI."""
    model_name = prompt_config.get('model', 'gemini-2.5-flash')
    temperature = prompt_config.get('temperature', 0.7)
    max_tokens = prompt_config.get('max_tokens', 8000)
    prompt_text = prompt_config.get('prompt', '')
    
    model = genai.GenerativeModel(model_name)
    
    generation_config = genai.types.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    
    all_results = []
    total_input_tokens = 0
    total_output_tokens = 0
    
    logger.info(f"Generating {num_batches} batch(es) using {model_name}...")
    logger.info(f"Prompt length: {len(prompt_text)} characters")
    
    for batch_num in range(num_batches):
        logger.info(f"Generating batch {batch_num + 1}/{num_batches}...")
        
        try:
            response = model.generate_content(
                prompt_text,
                generation_config=generation_config
            )
            
            # Log token usage if available
            if hasattr(response, 'usage_metadata'):
                usage = response.usage_metadata
                input_tokens = usage.prompt_token_count
                output_tokens = usage.candidates_token_count
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                logger.info(f"Batch {batch_num + 1} token usage: {input_tokens} input, {output_tokens} output")
            else:
                logger.debug("Token usage metadata not available in response")
            
            if response.text:
                try:
                    result = repair_json(response.text)
                    result = json.loads(result)
                except Exception as e:
                    logger.warning(f"Could not repair JSON response: {e}")
                    logger.debug(f"Raw response: {response.text[:200]}...")
                    continue
                if 'examples' in result:
                    all_results.extend(result['examples'])
                    logger.info(f"Generated {len(result['examples'])} examples")
                    
                    # Save checkpoint after each successful batch
                    if prompt_name:
                        checkpoint_token_usage = {
                            'input_tokens': total_input_tokens,
                            'output_tokens': total_output_tokens,
                            'total_tokens': total_input_tokens + total_output_tokens
                        }
                        checkpoint_file = f"data/{prompt_name}_checkpoint_batch_{batch_num + 1}.json"
                        save_results(all_results, checkpoint_file, prompt_name, checkpoint_token_usage)
                        logger.info(f"Checkpoint saved: {checkpoint_file}")
                        
            else:
                logger.warning("Empty response from model")
                
        except Exception as e:
            logger.error(f"Error generating content: {e}")
            continue
    
    # Log total token usage
    token_usage = {
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'total_tokens': total_input_tokens + total_output_tokens
    }
    
    if total_input_tokens > 0 or total_output_tokens > 0:
        logger.info(f"Total token usage: {total_input_tokens} input tokens, {total_output_tokens} output tokens")
        logger.info(f"Total tokens: {total_input_tokens + total_output_tokens}")
    
    return all_results, token_usage

def save_results(results: List[Dict[str, Any]], output_file: str, prompt_name: str, token_usage: Dict[str, int] = None) -> None:
    """Save generated results to JSON file."""
    output_data = {
        'prompt_name': prompt_name,
        'generated_at': datetime.now().isoformat(),
        'total_examples': len(results),
        'examples': results
    }
    
    if token_usage:
        output_data['token_usage'] = token_usage
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to: {output_file}")
        logger.info(f"Total examples generated: {len(results)}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary of generated results."""
    if not results:
        logger.warning("No results generated.")
        return
    
    domain_counts = {}
    for result in results:
        domain = result.get('domain', 'unknown')
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
    
    logger.info("Summary:")
    logger.info(f"Total examples: {len(results)}")
    logger.info("Domain distribution:")
    for domain, count in sorted(domain_counts.items()):
        logger.info(f"  {domain}: {count}")

def main(prompt_file: str, num_batches: int, summary: bool = True):
    prompt_file = Path(prompt_file)
    if not prompt_file.exists():
        logger.error(f"Prompt file '{prompt_file}' not found.")
        sys.exit(1)
    prompt_config = load_prompt_config(prompt_file)
    prompt_name = prompt_config.get('name', prompt_file.stem)
    setup_genai(os.getenv('GOOGLE_API_KEY'))
    results, token_usage = generate_content(prompt_config, num_batches, prompt_name)
    
    if not results:
        logger.error("No results generated. Exiting.")
        sys.exit(1)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f"data/{prompt_name}_generated_{timestamp}.json"    
    save_results(results, output_file, prompt_name, token_usage)
    print_summary(results)
    return None

if __name__ == '__main__':
    prompt_file = 'src/generator/prompt_normal_parsing.yaml'
    num_batches = 2000
    summary = True
    main(prompt_file, num_batches, summary)
