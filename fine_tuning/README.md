# Fine-Tuning Llama for Robot Command Interpretation

This directory contains a fine-tuning script for training a Llama model to interpret natural language robot commands and convert them into structured formats.

## Overview

The `finetune-llama.py` script uses the Unsloth library to efficiently fine-tune a DeepSeek R1 Distill Llama 8B model for robot command interpretation tasks. The training process uses LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning.

## Features

- **Memory Efficient**: Uses 4-bit quantization and LoRA for reduced memory usage
- **GPU Optimized**: Configured for CUDA GPU training with memory monitoring
- **Chat Template**: Implements Llama 3.1 chat template for proper conversation formatting
- **Response-Only Training**: Only computes loss on assistant responses for efficient learning
- **Multiple Export Formats**: Supports saving as LoRA adapter and GGUF format

## Requirements

### Dependencies

```bash
pip install unsloth
```

### Hardware Requirements

- CUDA-compatible GPU (configured for GPU #1 by default)
- Minimum 8GB GPU memory recommended
- The script automatically detects optimal dtype (Float16/Bfloat16)

## Dataset Format

The script expects a JSON file named `nlp-function-dataset.json` with the following structure:

```json
[
  {
    "string_cmd": "get a sponge from the pantry and deliver it to Jane in the living room",
    "structured_cmd": [
      {
        "action": "go_to",
        "location_to_go": "pantry"
      },
      {
        "action": "pick_object",
        "object_to_pick": "sponge"
      },
      {
        "action": "go_to",
        "location_to_go": "living room"
      },
      {
        "action": "find_person_by_name",
        "name": "Jane"
      },
      {
        "action": "give_object"
      }
    ]
  }
]
```

## Configuration

### Model Configuration

- **Base Model**: `unsloth/DeepSeek-R1-Distill-Llama-8B`
- **Max Sequence Length**: 2048 tokens
- **Quantization**: 4-bit enabled for memory efficiency

### LoRA Parameters

- **Rank (r)**: 16 - Controls adaptation capacity
- **Alpha**: 32 - LoRA scaling parameter
- **Target Modules**: Attention and MLP layers
- **Dropout**: 0 (optimized setting)

### Training Parameters

- **Batch Size**: 2 per device
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 2e-4
- **Epochs**: 3
- **Optimizer**: AdamW 8-bit
- **Scheduler**: Linear

## Usage

### 1. Prepare Your Dataset

Create a `nlp-function-dataset.json` file in the same directory with your training examples:

```json
[
  {
    "string_cmd": "Your natural language command",
    "structured_cmd": "The expected output  as a list of actions"
  }
]
```

### 2. Configure GPU

The script is configured to use GPU #1. To change this, modify the `CUDA_VISIBLE_DEVICES` setting:

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU #0 instead
```

### 3. Run Training

```bash
cd fine-tuning
python finetune-llama.py
```

### 4. Monitor Training

The script provides detailed memory usage statistics and training metrics:

- GPU memory usage before and after training
- Training time and performance metrics
- Peak memory consumption percentages

## Output Files

After training, the script generates several outputs:

### LoRA Adapter

- **Directory**: `lora_model/`
- **Contents**: LoRA adapter weights and tokenizer configuration
- **Use Case**: For loading and merging with base model

### GGUF Format Models

- **16-bit Float**: `model/llama/f16/`
- **4-bit Quantized**: `model/llama/q4/`
- **Use Case**: For deployment with llama.cpp, ollama, or similar tools

## Testing and Inference

The script includes built-in testing for three scenarios:

### 1. Robot Command Interpretation

Tests the primary training objective with robot commands:

```python
"Go to the kitchen, grab cookies and place them in the living room"
```

## Customization

### Changing the Base Model

Replace the model name in the configuration:

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit",  # Different base model
    # ... other parameters
)
```

### Adjusting LoRA Parameters

Modify the LoRA configuration for different adaptation strengths:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,  # Higher rank for more parameters
    lora_alpha = 64,  # Adjust scaling
    # ... other parameters
)
```

### Training Parameters

Adjust training settings in the `TrainingArguments`:

```python
args = TrainingArguments(
    per_device_train_batch_size = 4,  # Increase if you have more GPU memory
    num_train_epochs = 5,  # More epochs for better convergence
    learning_rate = 1e-4,  # Lower learning rate for stability
    # ... other parameters
)
```

## Performance Optimization

### Memory Usage

- The script uses 4-bit quantization to reduce memory usage
- LoRA adaptation minimizes trainable parameters
- Gradient checkpointing further reduces memory requirements

### Speed Optimization

- Unsloth library provides 2x speedup over standard implementations
- 8-bit optimizers reduce memory and computation overhead
- Efficient data loading with multiple processes

## Troubleshooting

### Out of Memory Errors

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

1. Verify CUDA installation: `nvidia-smi`
2. Check PyTorch CUDA support: `torch.cuda.is_available()`
3. Adjust `CUDA_VISIBLE_DEVICES` for correct GPU selection

## Model Architecture

The fine-tuned model follows this conversation structure:

```
System: You are a command interpreter for a robot...
User: [Natural language command]
Assistant: [Structured command in JSON format]
```
