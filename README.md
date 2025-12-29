<p align="center">
    <a href="https://huggingface.co/spaces/afr2903/frida-cortex"><img width="93" height="30" alt="HF Space" src="https://github.com/user-attachments/assets/bc5fa4e9-cee7-45f0-8c4f-9cc3e836b19d"></a>
</p>

# FRIDA Natural Language Command Interpreter

FRIDA is a natural language command interpreter for robotics that converts human instructions into structured robot actions. It uses LLM-powered parsing with BAML (boundaryml.com) to reliably interpret commands and generate executable task plans.

## Springer LNAI Paper at MICAI 2025

<p align="center">
    <a href="https://doi.org/10.1007/978-3-032-09037-9_24"><img width="400" height="100" src="https://micai.org/2025/wp-content/uploads/2025/06/LOGOS_header-1.png" alt="DOI"></a>
</p>

> **Taming the LLM: Reliable Task Planning for Robotics Using Parsing and Grounding**

<img width="680" height="565" alt="architecture_600dpi" src="https://github.com/user-attachments/assets/5aaf08fe-dbe5-4049-a879-340ff1818ec8" />

DOI: [10.1007/978-3-032-09037-9_24](https://doi.org/10.1007/978-3-032-09037-9_24)

### Supplementary Material

- [Explanatory video demonstration of task planning test using the command interpreter with the robot](https://www.youtube.com/watch?v=do1S1zfmMsA)
- [Video demonstration of the GPSR task during the Mexican Robotics Tournament 2025](https://youtube.com/shorts/0bMz6ESv6B8)
- [Video demonstration of the GPSR task during the RoboCup Competition 2025](https://youtu.be/mR20gFp2lA0)
- [Code repository of the complete deployed code of the robot for all tasks of the competition](https://github.com/RoBorregos/home2)

## Try It Online

**Recommended**: Use the [🤗 Space](https://huggingface.co/spaces/afr2903/frida-cortex) for the easiest way to try FRIDA without any setup.

Alternatively, there's another [Playground](https://frida-cortex.vercel.app/) online.

## Core Functionality

FRIDA interprets natural language commands and converts them into structured action sequences. For example:

**Input**: "Get a sponge from the pantry and deliver it to Jane in the living room"

**Output**:
```json
{
  "commands": [
    {"action": "go_to", "location_to_go": "pantry"},
    {"action": "pick_object", "object_to_pick": "sponge"},
    {"action": "go_to", "location_to_go": "living room"},
    {"action": "find_person_by_name", "name": "Jane"},
    {"action": "give_object"}
  ]
}
```

## Installation

### Prerequisites

- Python 3.8+
- Docker (for local model inference)
- Git (for submodules)

### Setup

1. **Clone the repository with submodules**:
```bash
git submodule update --init --recursive --remote
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up CommandGenerator** (required for command generation):
```bash
cd dataset_generator/CommandGenerator
python -m venv venv
source venv/bin/activate
pip install .
athome-generator -d ../CompetitionTemplate
cd ../..
```

4. **Generate BAML client**:
```bash
baml-cli generate --from command_interpreter/baml_src
```

5. **Configure environment** (for API-based models):
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Usage

### Command Line Interpreter

Run the interactive command interpreter:

```bash
python3 command_interpreter/interpreter.py
```

The interpreter supports:
- Natural language command input
- Model selection (press `m`)
- Random command generation (press `g`)
- Interactive execution of commands

### Local Model Inference

FRIDA supports running a fine-tuned local model using Ollama. This allows you to use the system without API keys.

#### 1. Download the Model

```bash
cd inference
./download-model.sh
```

This script:
- Downloads the fine-tuned model from Hugging Face
- Detects available Docker images
- Creates the Ollama model configuration

#### 2. Run the Inference Server

```bash
cd inference
./run-inference.sh
```

The script automatically:
- Detects your hardware (NVIDIA GPU, Apple Silicon, CPU)
- Configures Docker for optimal performance
- Starts the Ollama service on `http://localhost:11434`

**Platform-specific behavior**:
- **macOS (Apple Silicon)**: Uses ARM64-optimized containers, CPU mode
- **macOS (Intel)**: Uses x86_64 containers with emulation, CPU mode
- **Linux (no GPU)**: CPU mode with host networking
- **Linux (NVIDIA GPU)**: GPU acceleration with host networking (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))
- **Windows**: Standard port mapping, GPU support via Docker Desktop WSL2 backend

#### 3. Use Local Model

Once the inference server is running, the `LOCAL_FINETUNED` model will be available in the command interpreter. The model is accessible at `http://localhost:11434`.

### Alternative: Manual Docker Compose

For CPU-only mode using Docker Compose:

```bash
docker compose up
```

## Web Application

The project includes a web interface for interacting with the command interpreter.

### Quick Start

1. **Install web dependencies**:
```bash
pip install -r app-requirements.txt
```

2. **Configure environment**:
```bash
cp .env.example .env
# Add your GOOGLE_API_KEY to .env
```

3. **Run the Flask application**:
```bash
python main.py
```

4. **Access the interface**:
Open `http://localhost:8080` in your browser.

For a more advanced Next.js frontend with FastAPI backend, see the [frontend/](frontend/) and [backend/](backend/) directories.

## Dataset Generation

Generate training datasets for fine-tuning:

```bash
cd dataset_generator/
python3 structured_generator.py
```

The dataset will be generated in `dataset_generator/dataset.json`.

## Project Structure

```
frida-cortex/
├── command_interpreter/     # Core command interpretation logic
│   ├── baml_src/           # BAML model definitions
│   ├── interpreter.py      # Main CLI interpreter
│   └── ...
├── dataset_generator/       # Dataset generation tools
├── fine_tuning/            # Model fine-tuning scripts
├── inference/              # Local model inference setup
├── backend/                # FastAPI backend service
├── frontend/               # Next.js frontend application
└── main.py                 # Flask web application
```

## Model Support

FRIDA supports multiple LLM providers:

- **Local Fine-tuned**: `LOCAL_FINETUNED` (requires local inference server)
- **Google**: `GEMINI_PRO_2_5`, `GEMINI_FLASH_2_5`
- **OpenAI**: `OPENAI_GPT_4_1_MINI`
- **Anthropic**: `ANTHROPIC_CLAUDE_SONNET_4`
- **Meta**: `META_LLAMA_3_3_8B_IT_FREE`, `META_LLAMA_3_3_70B`

## Documentation

- [Fine-tuning Guide](fine_tuning/README.md) - How to fine-tune models for robot command interpretation
- [Backend API](backend/README.md) - FastAPI backend documentation
- [Frontend](frontend/README.md) - Next.js frontend documentation

## License

See [LICENSE](LICENSE) file for details.
