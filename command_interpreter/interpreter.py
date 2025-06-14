from baml_client.sync_client import b
from dotenv import load_dotenv
from baml_client.types import CommandListLLM
from baml_client.config import set_log_level
from caller import execute_function

set_log_level("ERROR")  # Set as "INFO" to see the full prompt and response
load_dotenv()

def print_commands_pretty(command_list):
    """Print the command list in a formatted, readable way"""
    print("\n" + "="*50)
    print("GENERATED COMMANDS")
    print("="*50)
    
    if hasattr(command_list, 'commands') and command_list.commands:
        for i, command in enumerate(command_list.commands, 1):
            print(f"\n{i}. {command.__class__.__name__}")
            
            # Use model_dump() to get clean field data without Pydantic internals
            if hasattr(command, 'model_dump'):
                fields = command.model_dump()
                for field_name, field_value in fields.items():
                    # Convert field name to readable format
                    readable_name = field_name.replace('_', ' ').title()
                    print(f"   {readable_name}: {field_value}")
            else:
                # Fallback for non-Pydantic objects
                print(f"   Action: {getattr(command, 'action', 'N/A')}")
    else:
        print("No commands generated.")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    print("Enter a command in natural language:")
    command = input()
    command_list = b.GenerateCommandListBaseModel(command)
    for cmd in command_list.commands:
        execute_function(cmd)
    # print_commands_pretty(command_list)
