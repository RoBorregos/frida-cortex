# =============================================================================
# GPU CONFIGURATION
# =============================================================================
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Set which GPU to use (GPU #1)

# =============================================================================
# MODEL SETUP AND CONFIGURATION
# =============================================================================
from unsloth import FastLanguageModel
import torch

# Model parameters
max_seq_length = 2048  # Maximum sequence length for training
dtype = None  # Auto-detect optimal dtype (Float16 for older GPUs, Bfloat16 for newer)
load_in_4bit = True  # Enable 4-bit quantization to reduce memory usage

# List of available pre-quantized 4-bit models for faster loading
fourbit_models = [
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",      # Llama-3.1 2x faster
    "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
    "unsloth/Meta-Llama-3.1-70B-bnb-4bit",
    "unsloth/Meta-Llama-3.1-405B-bnb-4bit",    # 4bit for 405b!
    "unsloth/Mistral-Small-Instruct-2409",     # Mistral 22b 2x faster!
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",           # Phi-3.5 2x faster!
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/gemma-2-9b-bnb-4bit",
    "unsloth/gemma-2-27b-bnb-4bit",            # Gemma 2x faster!

    "unsloth/Llama-3.2-1B-bnb-4bit",           # NEW! Llama 3.2 models
    "unsloth/Llama-3.2-1B-Instruct-bnb-4bit",
    "unsloth/Llama-3.2-3B-bnb-4bit",
    "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
] # More models at https://huggingface.co/unsloth

# Load the base model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/DeepSeek-R1-Distill-Llama-8B",  # Base model to fine-tune
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # Uncomment if using gated models requiring HuggingFace token
)

# =============================================================================
# LORA (LOW-RANK ADAPTATION) CONFIGURATION
# =============================================================================
# Configure LoRA parameters for efficient fine-tuning
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # LoRA rank - higher values = more parameters but better adaptation
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
                      "gate_proj", "up_proj", "down_proj",],   # MLP layers
    lora_alpha = 32,  # LoRA scaling parameter
    lora_dropout = 0,  # Dropout for LoRA layers (0 is optimized)
    bias = "none",     # Bias handling ("none" is optimized)
    use_gradient_checkpointing = "unsloth",  # Memory optimization technique
    random_state = 3407,  # Seed for reproducibility
    use_rslora = False,   # Rank stabilized LoRA
    loftq_config = None,  # LoftQ configuration
)

# =============================================================================
# CHAT TEMPLATE SETUP
# =============================================================================
from unsloth.chat_templates import get_chat_template

# Apply Llama 3.1 chat template for proper conversation formatting
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)

# =============================================================================
# DATA FORMATTING FUNCTIONS
# =============================================================================
def formatting_prompts_func(examples):
    """Convert conversation examples to text format using chat template"""
    convos = examples["conversations"]
    texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False) for convo in convos]
    return { "text" : texts, }

# =============================================================================
# DATASET LOADING AND PREPARATION
# =============================================================================
from datasets import Dataset
import json

def get_base_msgs():
    """Return the base system message for robot command interpretation"""
    return [
        {
            "role": "system",
            "content": """You are a command interpreter for a robot. Your task is to interpret the user's command and convert it into a structured format that the robot can understand."""
        },
    ]

def get_messages(msgs, text_input):
    """Add user message to conversation"""
    msgs.append({"role": "user", "content": text_input})
    return msgs

def get_response(msgs, response):
    """Add assistant response to conversation"""
    msgs.append({"role": "assistant", "content": response})
    return msgs

def load_custom_dataset(file_path):
    """Load and format custom dataset from JSON file"""
    res = []
    with open(file_path, "r") as f:
        data = json.load(f)

        # Convert each training example to conversation format
        for i in data:
            str_cmd = str(i["string_cmd"])    # Natural language command
            resp = str(i["structured_cmd"])   # Structured response
            
            # Create conversation: system message + user command + assistant response
            asda = get_base_msgs()
            a = get_response(get_messages(asda, str_cmd), resp)
            res.append(a)

    # Create HuggingFace dataset
    dataset_dict = {'conversations': res}
    dataset = Dataset.from_dict(dataset_dict)
    return dataset

# Load custom training dataset
dataset = load_custom_dataset('nlp-function-dataset.json')

# =============================================================================
# DATASET PREPROCESSING
# =============================================================================
from unsloth.chat_templates import standardize_sharegpt

# Standardize dataset format and apply text formatting
dataset = standardize_sharegpt(dataset)
dataset = dataset.map(formatting_prompts_func, batched = True,)

# =============================================================================
# TRAINING CONFIGURATION
# =============================================================================
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from unsloth import is_bfloat16_supported

# Initialize the supervised fine-tuning trainer
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    dataset_num_proc = 2,  # Number of processes for dataset processing
    packing = False,  # Whether to pack multiple sequences together
    args = TrainingArguments(
        per_device_train_batch_size = 2,  # Batch size per GPU
        gradient_accumulation_steps = 4,  # Steps to accumulate gradients
        warmup_steps = 5,  # Learning rate warmup steps
        num_train_epochs = 3,  # Number of training epochs
        learning_rate = 2e-4,  # Learning rate
        fp16 = not is_bfloat16_supported(),  # Use fp16 if bf16 not supported
        bf16 = is_bfloat16_supported(),      # Use bf16 if supported
        logging_steps = 1,  # Log every N steps
        optim = "adamw_8bit",  # Optimizer with 8-bit precision
        weight_decay = 0.01,   # Weight decay for regularization
        lr_scheduler_type = "linear",  # Learning rate scheduler
        seed = 3407,  # Random seed
        output_dir = "outputs",  # Directory to save outputs
        report_to = "none",  # Disable wandb/tensorboard logging
    ),
)

# =============================================================================
# RESPONSE-ONLY TRAINING SETUP
# =============================================================================
from unsloth.chat_templates import train_on_responses_only

# Configure trainer to only compute loss on assistant responses
trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
)

# =============================================================================
# TRAINING DATA INSPECTION
# =============================================================================
# Display sample of training data to verify formatting
tokenizer.decode(trainer.train_dataset[5]["input_ids"])

space = tokenizer(" ", add_special_tokens = False).input_ids[0]
tokenizer.decode([space if x == -100 else x for x in trainer.train_dataset[5]["labels"]])

# =============================================================================
# MEMORY MONITORING - PRE-TRAINING
# =============================================================================
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# =============================================================================
# TRAINING EXECUTION
# =============================================================================
trainer_stats = trainer.train()

# =============================================================================
# MEMORY MONITORING - POST-TRAINING
# =============================================================================
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory         /max_memory*100, 3)
lora_percentage = round(used_memory_for_lora/max_memory*100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# =============================================================================
# INFERENCE TESTING - ROBOT COMMAND INTERPRETATION
# =============================================================================
from unsloth.chat_templates import get_chat_template

# Re-apply chat template for inference
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "llama-3.1",
)
FastLanguageModel.for_inference(model)  # Optimize model for inference

# Test the fine-tuned model on robot command interpretation
messages = [
    {
        "content": """You are a command interpreter for a robot. Your task is to interpret the user's command and convert it into a structured format that the robot can understand.""",
        "role": "system"
    },
    { "content": "Go to the kitchen, grab cookies and place them in the living room", "role": "user" },
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

# Generate response
outputs = model.generate(input_ids = inputs, max_new_tokens = 64, use_cache = True,
                         temperature = 1.5, min_p = 0.1)
tokenizer.batch_decode(outputs)

# =============================================================================
# INFERENCE TESTING - GENERAL KNOWLEDGE (FIBONACCI)
# =============================================================================
FastLanguageModel.for_inference(model)

# Test general capabilities with Fibonacci sequence
messages = [
    {"role": "user", "content": "Continue the fibonnaci sequence: 1, 1, 2, 3, 5, 8,"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)

# =============================================================================
# MODEL SAVING - LORA ADAPTER
# =============================================================================
model.save_pretrained("lora_model")    # Save LoRA adapter weights
tokenizer.save_pretrained("lora_model")  # Save tokenizer configuration

# =============================================================================
# INFERENCE TESTING - CREATIVE DESCRIPTION
# =============================================================================
# Test creative capabilities
messages = [
    {"role": "user", "content": "Describe a tall tower in the capital of France."},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer, skip_prompt = True)
_ = model.generate(input_ids = inputs, streamer = text_streamer, max_new_tokens = 128,
                   use_cache = True, temperature = 1.5, min_p = 0.1)

# =============================================================================
# MODEL SAVING - GGUF FORMAT
# =============================================================================
# Save model in GGUF format for different quantization levels
if True: model.save_pretrained_gguf("model/llama/f16", tokenizer, quantization_method = "f16")      # 16-bit float
if True: model.save_pretrained_gguf("model/llama/q4", tokenizer, quantization_method = "q4_k_m")   # 4-bit quantized

