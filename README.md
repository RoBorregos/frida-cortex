# frida-cortex

FRIDA natural language command interpreter

## Execute Robocup's AtHome command generator

```bash

# Run in the root directory of the repository

cd dataset_generator/CommandGenerator
python -m venv venv
source venv/bin/activate
pip install .
athome-generator -d ../CompetitionTemplate
```
