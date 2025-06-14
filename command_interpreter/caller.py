
from tasks import Tasks

def search_command(command, objects: list[object]):
    for object in objects:
        if hasattr(object, command):
            method = getattr(object, command)
            if callable(method):
                return method
    return None

def execute_function(command):
    tasks = Tasks()
    try:
        exec_commad = search_command(
            command.action,
            [tasks],
        )
        if exec_commad is None:
            print(
                f"Command {command} is not implemented in GPSRTask or in the subtask managers."
            )
        else:
            status, res = exec_commad(command)
            print(f"command.action-> {str(command.action)}")
            print(f"status-> {str(status)}")
            print(f"res-> {str(res)}")

            try:
                status = status.value
            except Exception:
                try:
                    status = int(status)
                except Exception:
                    pass
                
                # TODO: command history
                # self.subtask_manager.hri.add_command_history(
                #     command,
                #     res,
                #     status,
                # )
    except Exception as e:
        print(
            f"Error occured while executing command ({str(command)}): " + str(e)
        )