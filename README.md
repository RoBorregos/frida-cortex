# frida-cortex

FRIDA natural language command interpreter

## Clone submodules

```bash
# Run in the root directory of the repository
git submodule update --init --recursive --remote
```

## Execute Robocup's AtHome command generator

First, clone the submodules if you haven't done so already. Then follow these steps to set up the command generator:

```bash
# Run in the root directory of the repository
cd dataset_generator/CommandGenerator
python -m venv venv
source venv/bin/activate
pip install .
athome-generator -d ../CompetitionTemplate
```

## Execute the dataset generator

First, clone the submodules if you haven't done so already. Then follow these steps to set up the dataset generator:

```bash
# Run in the root directory of the repository
pip install -r requirements.txt
cd dataset_generator/
python3 structured_generator.py

# The dataset will be generated in `dataset_generator/dataset.json`
```

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