# frida-cortex
FRIDA natural language command interpreter


## Execute Command Interpreter

If using an API based model, populate the `.env` file with your API keys, based on `.env.example`.

1. Generate BAML client

```bash
# Run in the root directory
pip install -r requirements.txt
baml-cli generate --from command_interpreter/baml_src
```
2. Run command interpreter

```bash
python3 command_interpreter/interpreter.py
```
You can enter a natural language command through text and the list of commands to be executed will be displayed.
