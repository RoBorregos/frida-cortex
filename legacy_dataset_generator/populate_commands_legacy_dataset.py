import json
import os
import sys
import time
from dotenv import load_dotenv
from tqdm import tqdm

# Add the parent directory to the Python path to enable absolute imports from baml_client

from baml_client.sync_client import b
from baml_client.config import set_log_level

# --- Configuration ---
DATASET_FILE = "dataset_legacy.json"
    
load_dotenv()
set_log_level("ERROR")  # Suppress detailed BAML logs


def enrich_dataset():
    """
    Reads the legacy dataset, generates structured commands using the specified BAML model,
    and updates the dataset file in place.
    """
    # Construct the full path to the dataset file
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), DATASET_FILE)

    # Load the dataset
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at '{dataset_path}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{dataset_path}'")
        return

    print(f"Starting to enrich {len(dataset)} commands...")

    # Process each command in the dataset
    for entry in tqdm(dataset, desc="Enriching commands"):
        # Process only if the structured_command is missing (null)
        if entry.get("structured_command") is None:
            command_text = entry["string_cmd"]
            time.sleep(1)
            try:
                # Generate the structured command using the fine-tuned model
                structured_response = b.GenerateCommandList(command_text)
                
                # Update the entry with the generated command, converted to a dictionary
                entry["structured_command"] = [cmd.model_dump() for cmd in structured_response.commands]

            except Exception as e:
                print(f"\nError processing command: '{command_text}' -> {e}")
                # Add an error message to the entry for easier debugging
                entry["structured_command"] = [{"error": str(e)}]

    # Save the updated dataset back to the same file
    try:
        with open(dataset_path, 'w') as f:
            json.dump(dataset, f, indent=4)
        print(f"\nSuccessfully enriched and saved dataset to '{dataset_path}'")
    except Exception as e:
        print(f"Error saving updated dataset: {e}")

if __name__ == "__main__":
    enrich_dataset() 