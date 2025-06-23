import subprocess
import os
import json
from pathlib import Path

def run_command(command, cwd):
    """Runs a command in a specified directory."""
    print(f"Running command: {' '.join(command)} in {cwd}")
    process = subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error running command: {' '.join(command)}")
        print(f"Stderr: {stderr}")
        raise RuntimeError(f"Command failed: {' '.join(command)}")
    print(stdout)
    return stdout

def main():
    # Path to the legacy generator source
    legacy_generator_path = Path(__file__).parent / 'legacy_generator_source'
    output_dataset_file = Path(__file__).parent / 'legacy_dataset.json'
    command_amount = 100

    # Build the project
    print("Building the legacy command generator...")
    run_command(['make'], cwd=legacy_generator_path)

    # Run the generators
    print("Running GPSR command generator...")
    run_command(['mono', 'bin/Release/GPSRCmdGen.exe', '--bulk', str(command_amount)], cwd=legacy_generator_path)

    # Collect the generated commands
    dataset = []
    grammar_dirs = [d for d in legacy_generator_path.iterdir() if d.is_dir() and d.name.startswith('GPSR')]

    for grammar_dir in grammar_dirs:
        for txt_file in grammar_dir.glob('*.txt'):
            generator_type = grammar_dir.name.split('_')[0]
            with open(txt_file, 'r') as f:
                for line in f:
                    # The file format is "ID. Command"
                    command_text = line.split('.', 1)[-1].strip()
                    if command_text:
                        dataset.append({
                            'type': generator_type,
                            'command': command_text
                        })
    
    # Save the dataset to a file
    with open(output_dataset_file, 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print(f"Dataset saved to {output_dataset_file}")


if __name__ == "__main__":
    main() 