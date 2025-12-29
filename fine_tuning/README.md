# Fine-Tuning Llama for Robot Command Interpretation

This directory contains scripts for fine-tuning a Llama model to interpret natural language robot commands and convert them into structured action sequences.

## Overview

The `finetune-llama.py` script uses Unsloth to efficiently fine-tune a DeepSeek R1 Distill Llama 8B model for robot command interpretation. Training uses LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning.

## Features

- 4-bit quantization and LoRA for reduced memory usage
- GPU-optimized training with automatic memory monitoring
- Llama 3.1 chat template for proper conversation formatting
- Response-only training (loss computed only on assistant responses)
- Multiple export formats (LoRA adapter and GGUF)

## Requirements

### Dependencies

```bash
pip install unsloth
```

### Hardware

- CUDA-compatible GPU (configured for GPU #1 by default)
- Minimum 8GB GPU memory recommended
- Script automatically detects optimal dtype (Float16/Bfloat16)

## Dataset Format

The script expects `nlp-function-dataset.json` with this structure:

```json
[
  {
    "string_cmd": "get a sponge from the pantry and deliver it to Jane in the living room",
    "structured_cmd": [
      {"action": "go_to", "location_to_go": "pantry"},
      {"action": "pick_object", "object_to_pick": "sponge"},
      {"action": "go_to", "location_to_go": "living room"},
      {"action": "find_person_by_name", "name": "Jane"},
      {"action": "give_object"}
    ]
  }
]
```

## Configuration

### Model

- **Base Model**: `unsloth/DeepSeek-R1-Distill-Llama-8B`
- **Max Sequence Length**: 2048 tokens
- **Quantization**: 4-bit enabled

### LoRA Parameters

- **Rank (r)**: 16
- **Alpha**: 32
- **Target Modules**: Attention and MLP layers
- **Dropout**: 0

### Training

- **Batch Size**: 2 per device
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 2e-4
- **Epochs**: 3
- **Optimizer**: AdamW 8-bit
- **Scheduler**: Linear

## Usage

### 1. Prepare Dataset

Create `nlp-function-dataset.json` in the same directory:

```json
[
  {
    "string_cmd": "Your natural language command",
    "structured_cmd": [{"action": "...", ...}]
  }
]
```

### 2. Configure GPU

The script uses GPU #1 by default. To change:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU #0
```

### 3. Run Training

```bash
cd fine_tuning
python finetune-llama.py
```

### 4. Monitor Training

The script provides:
- GPU memory usage before and after training
- Training time and performance metrics
- Peak memory consumption percentages

## Output Files

### LoRA Adapter

- **Directory**: `lora_model/`
- **Contents**: LoRA adapter weights and tokenizer configuration
- **Use Case**: For loading and merging with base model

### GGUF Format Models

- **16-bit Float**: `model/llama/f16/`
- **4-bit Quantized**: `model/llama/q4/`
- **Use Case**: For deployment with llama.cpp, ollama, or similar tools

## Customization

### Change Base Model

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    # ... other parameters
)
```

### Adjust LoRA Parameters

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,  # Higher rank for more parameters
    lora_alpha = 64,  # Adjust scaling
    # ... other parameters
)
```

### Training Parameters

```python
args = TrainingArguments(
    per_device_train_batch_size = 4,  # Increase if more GPU memory
    num_train_epochs = 5,  # More epochs
    learning_rate = 1e-4,  # Lower learning rate
    # ... other parameters
)
```

## Troubleshooting

### Out of Memory

1. Reduce `per_device_train_batch_size`
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce `max_seq_length`
4. Lower LoRA rank (`r` parameter)

### Poor Performance

1. Increase training epochs
2. Adjust learning rate
3. Increase LoRA rank for more adaptation capacity
4. Check dataset quality and formatting

### GPU Issues

1. Verify CUDA: `nvidia-smi`
2. Check PyTorch CUDA: `torch.cuda.is_available()`
3. Adjust `CUDA_VISIBLE_DEVICES` for correct GPU selection

## Model Architecture

The fine-tuned model uses this conversation structure:

```
System: You are a command interpreter for a robot...
User: [Natural language command]
Assistant: [Structured command in JSON format]
```
