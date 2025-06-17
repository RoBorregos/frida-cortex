"""Dataset enricher using BAML for enriched and reordered commands"""

import json
import time
import sys
import os
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Add path to import BAML client from parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'command_interpreter'))

try:
    from baml_client.sync_client import b
    from baml_client.config import set_log_level
except ImportError as e:
    print(f"Error importing BAML client: {e}")
    print("Make sure you're running this from the correct directory and BAML is properly set up.")
    sys.exit(1)

# Turn off BAML logging
set_log_level("ERROR")

# Configuration
ORIGINAL_DATASET_FILE = "dataset.json"
ENRICHED_DATASET_FILE = "dataset_enriched_reordered.json"
DEFAULT_MODEL = "GEMINI_FLASH_2_5"
DELAY_BETWEEN_CALLS = 2  # seconds to avoid rate limiting

def read_original_dataset(file_path: str):
    """Read the original dataset from JSON file"""
    try:
        with open(file_path, 'r') as f:
            dataset = json.load(f)
        print(f"Successfully loaded {len(dataset)} commands from {file_path}")
        return dataset
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None


def enrich_and_reorder_command(original_command: str) -> str:
    """Use BAML to enrich and reorder a single command"""
    try:
        enriched_command = b.GenerateEnrichedAndReorderedCommand(original_command)
        return enriched_command
    except Exception as e:
        print(f"Warning: Failed to enrich command '{original_command}': {e}")
        return original_command  # Return original if enrichment fails


def process_dataset(original_dataset):
    """Process the entire dataset, enriching and reordering string commands"""
    enriched_dataset = []
    
    print(f"Processing {len(original_dataset)} commands...")
    
    for i, command_data in enumerate(tqdm(original_dataset, desc="Enriching commands")):
        try:
            # Create a copy of the original command data
            enriched_command_data = command_data.copy()
            
            # Get the original string command
            original_string_cmd = command_data.get('string_cmd', '')
            
            if not original_string_cmd:
                print(f"Warning: No string_cmd found for command {i}, skipping enrichment")
                enriched_dataset.append(enriched_command_data)
                continue
            
            # Enrich and reorder the command
            print(f"\nProcessing command {i+1}/{len(original_dataset)}")
            print(f"Original: {original_string_cmd}")
            
            enriched_string_cmd = enrich_and_reorder_command(original_string_cmd)
            print(f"Enriched: {enriched_string_cmd}")
            
            # Update the string_cmd with the enriched version
            enriched_command_data['string_cmd'] = enriched_string_cmd
            
            # Add metadata about the enrichment
            enriched_command_data['original_string_cmd'] = original_string_cmd
            enriched_command_data['enrichment_applied'] = True
            
            enriched_dataset.append(enriched_command_data)
            
            # Add delay to avoid rate limiting
            time.sleep(DELAY_BETWEEN_CALLS)
                
        except KeyboardInterrupt:
            print(f"\nInterrupted by user. Processed {i} commands so far.")
            save_partial_results(enriched_dataset, i)
            return None
        except Exception as e:
            print(f"Error processing command {i}: {e}")
            # Add the original command without enrichment
            enriched_command_data = command_data.copy()
            enriched_command_data['enrichment_applied'] = False
            enriched_command_data['enrichment_error'] = str(e)
            enriched_dataset.append(enriched_command_data)
    
    return enriched_dataset


def save_enriched_dataset(enriched_dataset, file_path: str):
    """Save the enriched dataset to JSON file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(enriched_dataset, f, indent=2)
        print(f"Successfully saved enriched dataset to {file_path}")
        return True
    except Exception as e:
        print(f"Error saving enriched dataset: {e}")
        return False


def save_partial_results(enriched_dataset, processed_count):
    """Save partial results in case of interruption"""
    partial_file = f"dataset_enriched_reordered_partial_{processed_count}.json"
    try:
        with open(partial_file, 'w') as f:
            json.dump(enriched_dataset, f, indent=2)
        print(f"Partial results saved to {partial_file}")
    except Exception as e:
        print(f"Error saving partial results: {e}")


def print_sample_comparison(original_dataset, enriched_dataset, num_samples=3):
    """Print a comparison of original vs enriched commands"""
    print(f"\n{'='*80}")
    print("SAMPLE COMPARISON")
    print(f"{'='*80}")
    
    sample_indices = range(min(num_samples, len(original_dataset), len(enriched_dataset)))
    
    for i in sample_indices:
        original_cmd = original_dataset[i].get('string_cmd', 'N/A')
        enriched_cmd = enriched_dataset[i].get('string_cmd', 'N/A')
        
        print(f"\nSample {i+1}:")
        print(f"Original:  {original_cmd}")
        print(f"Enriched:  {enriched_cmd}")
        print("-" * 80)


def main():
    """Main execution function"""
    print("Dataset Enricher - Enriching and Reordering Commands")
    print(f"{'='*60}")
    
    print(f"Using model: {DEFAULT_MODEL}")
    
    # Read original dataset
    print(f"Reading original dataset from: {ORIGINAL_DATASET_FILE}")
    original_dataset = read_original_dataset(ORIGINAL_DATASET_FILE)
    
    if not original_dataset:
        print("Exiting due to dataset loading error.")
        return
    
    # Ask user for confirmation
    print(f"\nThis will process {len(original_dataset)} commands.")
    print(f"Estimated time: ~{(len(original_dataset) * DELAY_BETWEEN_CALLS) / 60:.1f} minutes")
    
    proceed = input("Do you want to proceed? (y/N): ").strip().lower()
    if proceed not in ['y', 'yes']:
        print("Operation cancelled by user.")
        return
    
    # Process the dataset
    enriched_dataset = process_dataset(original_dataset)
    
    if enriched_dataset is None:
        print("Processing was interrupted.")
        return
    
    # Save enriched dataset
    print(f"\nSaving enriched dataset to: {ENRICHED_DATASET_FILE}")
    if save_enriched_dataset(enriched_dataset, ENRICHED_DATASET_FILE):
        print("Dataset enrichment completed successfully!")
        
        # Print statistics
        enriched_count = sum(1 for cmd in enriched_dataset if cmd.get('enrichment_applied', False))
        error_count = len(enriched_dataset) - enriched_count
        
        print(f"\nStatistics:")
        print(f"Total commands: {len(enriched_dataset)}")
        print(f"Successfully enriched: {enriched_count}")
        print(f"Errors/Skipped: {error_count}")
        print(f"Success rate: {(enriched_count/len(enriched_dataset)*100):.1f}%")
        
        # Show sample comparison
        print_sample_comparison(original_dataset, enriched_dataset)
        
    else:
        print("Failed to save enriched dataset.")


if __name__ == "__main__":
    main() 