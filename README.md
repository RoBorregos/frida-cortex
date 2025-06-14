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

### Executing with local LLM

1. Download the model

```bash
# In the project root directory
cd inference
./download-model.sh
```

2. Run the container

```bash
# In the project root directory
cd inference
# This script automatically detects your hardware and uses GPU acceleration if available
./run-inference.sh
```

Now, the model can be reached at `http://localhost:11434`.

### Platform-specific notes

The configuration is designed to be multi-platform and should work out of the box:

- **macOS (Apple Silicon):** The application will automatically use ARM64-optimized containers for better performance on M1/M2/M3 chips. CPU-only mode is used.
- **macOS (Intel):** The application will run on CPU using x86_64 containers with emulation.
- **Linux (without NVIDIA GPU):** The application will run on CPU and use host networking for optimal performance.
- **Linux (with NVIDIA GPU):** If you have an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed, the scripts will automatically detect and use your GPU for hardware acceleration with host networking.
- **Windows (Git Bash/WSL):** Run scripts from Git Bash or WSL terminal. GPU acceleration is supported with NVIDIA cards via Docker Desktop's WSL2 backend. Uses standard port mapping.

The `run-inference.sh` script provides:
- ✅ **Automatic hardware detection** - detects NVIDIA GPU and Docker runtime availability
- ✅ **Graceful fallback** - uses CPU mode if GPU acceleration is not available
- ✅ **Clear feedback** - shows exactly what hardware configuration is being used
- ✅ **No manual configuration** - works out of the box on any platform

### Alternative: Manual Docker Compose

If you prefer to use Docker Compose directly (CPU-only mode):

```bash
docker compose up
```
