from command_interpreter.tasks import Tasks
from termcolor import colored

def search_command(command, objects: list[object]):
    for object in objects:
        if hasattr(object, command):
            method = getattr(object, command)
            if callable(method):
                return method
    return None

def execute_function(command, tasks, grounding: bool = True):
    try:
        exec_commad = search_command(
            command.action,
            [tasks],
        )
        if exec_commad is None:
            print(colored(f"❌ Command {command} is not implemented in GPSRTask or in the subtask managers.", "red"))
        else:
            # Enhanced execution output with better formatting
            print(f"\n{colored('▶️  EXECUTING:', 'green', attrs=['bold'])} {colored(command.action, 'cyan', attrs=['bold'])}")
            print(colored("─" * 60, "green"))
            
            status, res = exec_commad(command, grounding)
            
            # Format and display results with better visual hierarchy
            print(f"{colored('🎯 Action:', 'yellow', attrs=['bold'])} {colored(str(command.action), 'white', attrs=['bold'])}")
            
            # Status with appropriate color based on success/failure
            try:
                status_value = status.value if hasattr(status, 'value') else int(status)
                status_str = str(status).upper()
                if status_value == 0 or 'EXECUTION_SUCCESS' in status_str or status_str == 'SUCCESS':
                    status_color = 'green'
                    status_icon = '✅'
                else:
                    status_color = 'red' 
                    status_icon = '❌'
            except Exception:
                status_color = 'yellow'
                status_icon = '⚠️'
            
            print(f"{colored('📊 Status:', 'yellow', attrs=['bold'])} {colored(status_icon, status_color)} {colored(str(status), status_color, attrs=['bold'])}")
            print(f"{colored('📝 Result:', 'yellow', attrs=['bold'])} {colored(str(res), 'cyan', attrs=['bold'])}")
            print(colored("─" * 60, "green"))
            print(colored("✓ Command execution completed", "green"))

            try:
                status = status.value
            except Exception:
                try:
                    status = int(status)
                except Exception:
                    pass
                
            tasks.add_command_history(
                command,
                res,
                status,
            )

            # return action (str), success (bool), result (str)
            return (command.action,
                    True if status_color == 'green' else False,
                    res)
    except Exception as e:
        print(colored("─" * 60, "red"))
        print(colored("💥 EXECUTION ERROR", "red", attrs=['bold']))
        print(colored("─" * 60, "red"))
        print(colored(f"Command: {str(command)}", "yellow"))
        print(colored(f"Error: {str(e)}", "red", attrs=['bold']))
        print(colored("─" * 60, "red"))

def clear_command_history(tasks):
    tasks.clear_command_history()