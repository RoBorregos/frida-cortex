import sys
import os
import re
import warnings

# Add the parent directory to Python path to enable absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baml_client.sync_client import b
from baml_py import ClientRegistry
from dotenv import load_dotenv
from baml_client.types import CommandListLLM
from baml_client.config import set_log_level
from command_interpreter.caller import execute_function
from termcolor import colored

# Add the dataset_generator directory to the path to import the CommandGenerator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dataset_generator'))
from gpsr_commands_structured import CommandGenerator

# Import parsing functions from the CommandGenerator submodule to avoid code duplication
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dataset_generator', 'CommandGenerator', 'src'))
try:
    from robocupathome_generator.generator import (
        read_data, parse_names, parse_locations, parse_rooms, parse_objects
    )
    GENERATOR_FUNCTIONS_AVAILABLE = True
except ImportError:
    print(colored("Warning: Could not import CommandGenerator parsing functions. Using fallback functions.", "yellow"))
    GENERATOR_FUNCTIONS_AVAILABLE = False

set_log_level("ERROR")  # Set as "INFO" to see the full prompt and response
load_dotenv()

# Available models from client.baml
AVAILABLE_MODELS = [
    'LOCAL_FINETUNED',
    'GEMINI_PRO_2_5',
    'GEMINI_FLASH_2_5',
    'OPENAI_GPT_4_1_MINI',
    'ANTHROPIC_CLAUDE_SONNET_4',
    'META_LLAMA_3_3_8B_IT_FREE',
    'GEMMA_2_7B'
]

DEFAULT_MODEL = 'GEMINI_FLASH_2_5'
current_model = DEFAULT_MODEL

# Initialize client registry
client_registry = ClientRegistry()
client_registry.set_primary(current_model)

def print_commands_pretty(command_list):
    """Print the command list in a formatted, readable way with colors"""
    print("\n" + colored("="*60, "blue"))
    print(colored("🧠 GENERATED COMMANDS", "blue", attrs=['bold']))
    print(colored("="*60, "blue"))
    
    if hasattr(command_list, 'commands') and command_list.commands:
        for i, command in enumerate(command_list.commands, 1):
            print(f"\n{colored(f'{i}.', 'cyan', attrs=['bold'])} {colored(command.__class__.__name__, 'magenta', attrs=['bold'])}")
            
            # Use model_dump() to get clean field data without Pydantic internals
            if hasattr(command, 'model_dump'):
                fields = command.model_dump()
                for field_name, field_value in fields.items():
                    # Convert field name to readable format
                    readable_name = field_name.replace('_', ' ').title()
                    print(f"   {colored(readable_name + ':', 'yellow', attrs=['bold'])} {colored(str(field_value), 'white', attrs=['bold'])}")
            else:
                # Fallback for non-Pydantic objects
                action_value = getattr(command, 'action', 'N/A')
                print(f"   {colored('Action:', 'yellow', attrs=['bold'])} {colored(action_value, 'white', attrs=['bold'])}")
    else:
        print(colored("❌ No commands generated.", "red", attrs=['bold']))
    
    print("\n" + colored("="*60, "blue"))
    print(colored("✅ Command generation completed!", "blue", attrs=['bold']))
    print(colored("="*60, "blue"))

def print_instructions():
    """Print the instructions for using the command interpreter"""
    print("\n" + colored("="*60, "magenta"))
    print(colored("FRIDA COMMAND INTERPRETER", "magenta", attrs=['bold']))
    print(colored("="*60, "magenta"))
    print(f"{colored('Current Model:', 'cyan', attrs=['bold'])} {colored(current_model, 'green', attrs=['bold'])}")
    print(f"\n{colored('Instructions:', 'yellow', attrs=['bold'])}")
    print(f"• {colored('Enter a natural language command', 'white')} to interpret it")
    print(f"• Type {colored('m', 'green', attrs=['bold'])} to select a different model")
    print(f"• Type {colored('g', 'green', attrs=['bold'])} to generate a random command and execute it")
    print(f"• Type {colored('q', 'red', attrs=['bold'])} to quit")
    print(colored("="*60, "magenta"))

def select_model():
    """Display model selection menu and allow user to choose"""
    global current_model
    
    print("\n" + colored("="*40, "cyan"))
    print(colored("MODEL SELECTION", "cyan", attrs=['bold']))
    print(colored("="*40, "cyan"))
    
    for i, model in enumerate(AVAILABLE_MODELS):
        if model == current_model:
            print(f"{colored('→', 'green', attrs=['bold'])} {colored(f'{i+1}.', 'green', attrs=['bold'])} {colored(model, 'green', attrs=['bold'])}")
        else:
            print(f"  {colored(f'{i+1}.', 'white')} {colored(model, 'white')}")
    
    print(f"\n{colored('Enter the number of the model you want to select:', 'yellow')}")
    
    try:
        choice = input(colored("Choice: ", "cyan")).strip()
        if choice.isdigit():
            choice_num = int(choice) - 1
            if 0 <= choice_num < len(AVAILABLE_MODELS):
                current_model = AVAILABLE_MODELS[choice_num]
                client_registry.set_primary(current_model)
                print(colored(f"\nModel changed to: {current_model}", "green", attrs=['bold']))
            else:
                print(colored("Invalid choice. Please select a number from the list.", "red"))
        else:
            print(colored("Invalid input. Please enter a number.", "red"))
    except KeyboardInterrupt:
        print(colored("\nReturning to main menu...", "yellow"))

def setup_command_generator():
    """Set up the CommandGenerator with data from CompetitionTemplate using reusable functions"""
    try:
        # File paths relative to the dataset_generator directory
        base_path = os.path.join(os.path.dirname(__file__), '..', 'dataset_generator')
        names_file_path = os.path.join(base_path, 'CompetitionTemplate', 'names', 'names.md')
        locations_file_path = os.path.join(base_path, 'CompetitionTemplate', 'maps', 'location_names.md')
        rooms_file_path = os.path.join(base_path, 'CompetitionTemplate', 'maps', 'room_names.md')
        objects_file_path = os.path.join(base_path, 'CompetitionTemplate', 'objects', 'objects.md')
        
        if GENERATOR_FUNCTIONS_AVAILABLE:
            # Use the reusable functions from the CommandGenerator submodule
            print(colored("Using CommandGenerator parsing functions...", "green"))
            
            names_data = read_data(names_file_path)
            names = parse_names(names_data)

            locations_data = read_data(locations_file_path)
            location_names, placement_location_names = parse_locations(locations_data)

            rooms_data = read_data(rooms_file_path)
            room_names = parse_rooms(rooms_data)

            objects_data = read_data(objects_file_path)
            object_names, object_categories_plural, object_categories_singular = parse_objects(objects_data)
        else:
            print(colored("Warning: Could not import CommandGenerator parsing functions.", "red"))
            sys.exit(1)

        # Create and return the CommandGenerator
        return CommandGenerator(names, location_names, placement_location_names, room_names, object_names,
                               object_categories_plural, object_categories_singular)
    
    except Exception as e:
        print(colored(f"Warning: Could not set up CommandGenerator: {e}", "red"))
        print(colored("Make sure the CompetitionTemplate submodule is initialized.", "yellow"))
        return None

def generate_and_execute_command(command_generator):
    """Generate a random command and execute it with the current model"""
    if command_generator is None:
        print(colored("Command generator is not available. Please check the CompetitionTemplate setup.", "red"))
        return
    
    try:
        # Generate a random command
        string_cmd, structured_cmd = command_generator.generate_full_command()
        
        print(f"\n{colored('='*50, 'green')}")
        print(colored("GENERATED COMMAND", "green", attrs=['bold']))
        print(colored("="*50, "green"))
        print(f"{colored('Command:', 'yellow', attrs=['bold'])} {colored(string_cmd, 'white', attrs=['bold'])}")
        print(colored("="*50, "green"))
        
        # Execute the generated command with the current model
        execute_command(string_cmd)
        
    except Exception as e:
        print(colored(f"Error generating command: {e}", "red"))

def execute_command(command_text):
    """Execute a command using the appropriate model function"""
    try:
        print(f"\n{colored('Executing with model:', 'cyan')} {colored(current_model, 'green', attrs=['bold'])}")
        
        if current_model == "LOCAL_FINETUNED":
            command_list = b.GenerateCommandListFineTuned(command_text,
                                                          baml_options={"client_registry": client_registry})
        else:
            command_list = b.GenerateCommandList(command_text,
                                                 baml_options={"client_registry": client_registry})
        
        print_commands_pretty(command_list)
        print("\n" + colored("="*60, "green"))
        print(colored("🚀 EXECUTING COMMANDS", "green", attrs=['bold']))
        print(colored("="*60, "green"))
        for cmd in command_list.commands:
            execute_function(cmd)
        print(colored("="*60, "green"))
        print(colored("🎉 All commands completed!", "green", attrs=['bold']))
        print(colored("="*60, "green"))
        
    except Exception as e:
        print(colored(f"Error executing command: {e}", "red"))

def main():
    """Main interactive loop"""
    global current_model
    
    # Set up command generator
    command_generator = None
    try:
        command_generator = setup_command_generator()
    except Exception as e:
        print(colored(f"Warning: Command generator setup failed: {e}", "red"))
    
    # Set initial model
    client_registry.set_primary(current_model)
    
    print(colored("Welcome to FRIDA Command Interpreter!", "magenta", attrs=['bold']))
    
    while True:
        print_instructions()
        
        try:
            user_input = input(f"\n{colored('Enter your command:', 'cyan')} ").strip()
            
            if user_input.lower() == 'q':
                print(colored("Goodbye!", "green", attrs=['bold']))
                break
            elif user_input.lower() == 'm':
                select_model()
            elif user_input.lower() == 'g':
                generate_and_execute_command(command_generator)
            elif user_input:
                # Natural language command
                execute_command(user_input)
            else:
                print(colored("Please enter a valid command.", "yellow"))
                
        except KeyboardInterrupt:
            print(colored("\n\nGoodbye!", "green", attrs=['bold']))
            break
        except Exception as e:
            print(colored(f"An error occurred: {e}", "red"))

if __name__ == "__main__":
    main()
