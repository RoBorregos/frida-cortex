"""Testing command interpreter using BAML for structured responses"""

import json
import time
from typing import Optional, List, Dict
from collections import defaultdict

# Updated imports to reference parent directory level
import sys
import os
from dotenv import load_dotenv
load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from baml_client.sync_client import b
from baml_client.types import CommandListLLM
from baml_client.config import set_log_level
from baml_py import ClientRegistry
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from tqdm import tqdm
import numpy as np
from pydantic import BaseModel

# Turn off all logging
set_log_level("ERROR")

# Available models (copied from interpreter.py)
AVAILABLE_MODELS = [
    "LOCAL_FINETUNED",
    "GEMINI_PRO_2_5",
    "GEMINI_FLASH_2_5",
    "OPENAI_GPT_4_1_MINI",
    "ANTHROPIC_CLAUDE_SONNET_4",
    "META_LLAMA_3_3_8B_IT_FREE",
    "META_LLAMA_3_3_70B"
]

DEFAULT_MODEL = "GEMINI_FLASH_2_5"

# Initialize client registry
client_registry = ClientRegistry()

# --- Configuration ---
STARTING_CASE = 0  # Adjust if needed
SIMILARITY_THRESHOLD = 0.8  # Threshold for complement similarity
OVERALL_THRESHOLD = 0.75  # Threshold for the overall test case score
TEST_DATA_FILE = "../../dataset_generator/dataset.json"
EMBEDDING_MODEL = "all-MiniLM-L12-v2"  # Example model

# --- Task Category Mapping ---
# A -> navigate to a location, look for a person, and follow
# B -> take an object from a placement, and perform an action  
# C -> Speak or answer a question

COMMAND_CATEGORY = {
    'A': ["followNameFromBeacToRoom", "guideNameFromBeacToBeac", "guidePrsFromBeacToBeac", "guideClothPrsFromBeacToBeac", "meetNameAtLocThenFindInRm", "followPrsAtLoc"],
    'A|B|C': ["goToLoc"],
    'A|C': ["findPrsInRoom", "meetPrsAtBeac", "greetClothDscInRm", "greetNameInRm"],
    'B': ["takeObjFromPlcmt", "findObjInRoom", "bringMeObjFromPlcmt"],
    'C': ["countObjOnPlcmt","countPrsInRoom", "tellPrsInfoInLoc", "tellObjPropOnPlcmt", "talkInfoToGestPrsInRoom", "answerToGestPrsInRoom", "tellCatPropOnPlcmt", "countClothPrsInRoom", "tellPrsInfoAtLocToPrsAtLoc"],
}

TASK_TYPE_DESCRIPTIONS = {
    'A': 'Navigate to a location, look for a person, and follow',
    'B': 'Take an object from a placement, and perform an action',
    'C': 'Speak or answer a question'
}

# --- Pydantic Models for Structured Results ---

class TestResult(BaseModel):
    """Individual test case result"""
    index: int
    input_command: str
    expected_command_count: int
    actual_command_count: int
    score: float
    passed: bool
    execution_time: float
    error: Optional[str] = None
    expected_commands: Optional[List[Dict]] = None
    actual_commands: Optional[List[Dict]] = None
    cmd_category: Optional[str] = None  # New field for task category


class CommandCountGroup(BaseModel):
    """Results grouped by command count"""
    command_count: int
    passed_count: int
    failed_count: int
    total_count: int
    pass_rate: float
    test_results: List[TestResult]


class TaskTypeGroup(BaseModel):
    """Results grouped by task type"""
    task_type: str
    task_description: str
    passed_count: int
    failed_count: int
    total_count: int
    pass_rate: float
    test_results: List[TestResult]


class TestSummary(BaseModel):
    """Complete test execution summary"""
    model_name: str
    total_cases: int
    total_passed: int
    total_failed: int
    overall_pass_rate: float
    average_execution_time: float
    groups_by_command_count: List[CommandCountGroup]
    groups_by_task_type: List[TaskTypeGroup]  # New field for task type grouping
    
    def print_summary(self):
        """Print a formatted summary of the test results"""
        print("\n" + "=" * 60)
        print(f"TEST SUMMARY - Model: {self.model_name}")
        print("=" * 60)
        print(f"Total Cases: {self.total_cases}")
        print(f"Passed: {self.total_passed} ({self.overall_pass_rate:.1f}%)")
        print(f"Failed: {self.total_failed}")
        print(f"Average Execution Time: {self.average_execution_time:.2f}s")
        
        print(f"\nResults by Command Count:")
        print("-" * 40)
        for group in sorted(self.groups_by_command_count, key=lambda x: x.command_count):
            print(f"Commands: {group.command_count:2d} | "
                  f"Total: {group.total_count:3d} | "
                  f"Passed: {group.passed_count:3d} | "
                  f"Failed: {group.failed_count:3d} | "
                  f"Pass Rate: {group.pass_rate:5.1f}%")
        
        print(f"\nResults by Task Type:")
        print("-" * 60)
        for group in sorted(self.groups_by_task_type, key=lambda x: x.task_type):
            print(f"Task {group.task_type}: {group.task_description}")
            print(f"  Total: {group.total_count:3d} | "
                  f"Passed: {group.passed_count:3d} | "
                  f"Failed: {group.failed_count:3d} | "
                  f"Pass Rate: {group.pass_rate:5.1f}%")
        print("=" * 60)


# --- Helper Functions ---


def parse_expected_output(json_string) -> Optional[CommandListLLM]:
    """Parses the JSON string from the dataset into an ExpectedCommandList model."""
    try:
        # Replace single quotes and None representation if necessary
        # Handle potential variations in JSON string format
        data = {"commands": json_string}
        return CommandListLLM(**data)
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        print(f"Error parsing JSON string: {json_string}\nError: {e}")
        return None


# --- Comparison Logic ---

# Load the embedding model globally
print(f"Loading embedding model: {EMBEDDING_MODEL}...")
try:
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded.")
except Exception as e:
    print(f"\x1b[91mError loading sentence transformer model '{EMBEDDING_MODEL}': {e}\x1b[0m")
    print("\x1b[93mPlease ensure 'sentence-transformers' and 'torch' (or 'tensorflow') are installed.\x1b[0m")
    print("Cosine similarity will default to exact string matching.")
    model = None


def calculate_cosine_similarity(text1: Optional[str], text2: Optional[str]) -> float:
    """Calculates cosine similarity between two texts using an embedding model.
    Returns 1.0 for perfect match (including both None), 0.0 if only one is None,
    and cosine similarity if both are strings and model is loaded, otherwise exact match.
    """
    if text1 is None and text2 is None:
        return 1.0
    if text1 is None or text2 is None:
        return 0.0
    # Now we know both text1 and text2 are strings
    if model:
        try:
            # Ensure texts are not empty strings, as encode might handle them differently
            if not text1 or not text2:
                return 1.0 if text1 == text2 else 0.0

            embeddings = model.encode([text1, text2])
            # Ensure embeddings are 2D numpy arrays for cosine distance
            if embeddings.ndim == 1:
                # This case might happen if encode returns a single vector for short/empty strings depending on the model
                # We should handle based on specific model behavior, but reshaping assumes two vectors were intended.
                embeddings = embeddings.reshape(2, -1)  # Adjust if model behaves differently

            # Check if embeddings have the expected shape (2, embedding_dim)
            if embeddings.shape[0] != 2:
                print(f"Warning: Unexpected embedding shape {embeddings.shape} for texts: '{text1}', '{text2}'")
                return 1.0 if text1 == text2 else 0.0  # Fallback to exact match

            # scipy.spatial.distance.cosine returns the distance (1 - similarity)
            similarity = 1 - cosine(embeddings[0], embeddings[1])
            # Handle potential NaN results from cosine if vectors are zero, etc.
            return float(np.nan_to_num(similarity))
        except Exception as e:
            print(f"Error calculating similarity for '{text1}' vs '{text2}': {e}")
            # Fallback to exact match on error
            return 1.0 if text1 == text2 else 0.0
    else:
        # Fallback to exact string match if model wasn't loaded
        print(f"WARN: Embedding model not loaded. Comparing '{text1}' vs '{text2}' as exact match.")
        return 1.0 if text1 == text2 else 0.0


def compare_commands(actual: CommandListLLM, expected: CommandListLLM) -> float:
    """Compares the actual BAML output with the expected output."""
    if len(actual.commands) != len(expected.commands):
        print(f"Mismatch in number of commands: Actual={len(actual.commands)}, Expected={len(expected.commands)}")
        return 0.0

    if not actual.commands:  # Handle empty command list case
        return 1.0

    scores = []
    for i, (act_cmd, exp_cmd) in enumerate(zip(actual.commands, expected.commands)):
        command_score = 0.0
        action_match = False
        similarity = 0.0

        # Compare Action (Enum vs String)
        # Convert Enum member to its value (string representation)
        if act_cmd.action == exp_cmd.action:
            action_match = True
            # Compare Complement using Cosine Similarity (handles None)
            similarity = calculate_cosine_similarity(str(act_cmd), str(exp_cmd))

            # Score for this command: Weighted average? Let's try 40% action, 60% complement
            # command_score = 0.4 + (0.6 * similarity) if similarity >= 0 else 0.0 # Ensure similarity isn't negative

            # Simpler: 1 if action matches and similarity is above threshold, 0 otherwise?
            # Let's stick to the average for now, gives more granular scoring.
            command_score = (1.0 + similarity) / 2.0  # Simple average of action (1) and similarity

        else:
            print(f"Command {i}: Action mismatch - Actual='{act_cmd.action}', Expected='{exp_cmd.action}'")
            command_score = 0.0  # Action mismatch means 0 score for the command

        scores.append(command_score)
        print(f"  Cmd {i}: Act='{act_cmd.action}' vs Exp='{exp_cmd.action}' -> ActionMatch={action_match}, Sim={similarity:.2f}, Score={command_score:.2f}")

    overall_score = np.mean(scores) if scores else 1.0
    return overall_score


# --- Main Test Execution ---

def execute_command_with_model(command_text: str, model_name: str):
    """Execute a command using the appropriate model function"""
    if model_name == "LOCAL_FINETUNED":
        return b.GenerateCommandListFineTuned(command_text,
                                              baml_options={"client_registry": client_registry})
    else:
        return b.GenerateCommandList(command_text,
                                     baml_options={"client_registry": client_registry})


def run_tests(model_name: str = DEFAULT_MODEL, use_semantic_enrichment: bool = False) -> TestSummary:
    """Loads data, runs tests, and returns structured results.
    
    Args:
        model_name: The model to use for testing. Must be one of AVAILABLE_MODELS.
        use_semantic_enrichment: If True, applies semantic enrichment to input commands before processing.
        
    Returns:
        TestSummary: Structured results including grouping by command count
    """
    # Validate model name
    if model_name not in AVAILABLE_MODELS:
        print(f"Error: Model '{model_name}' not available. Available models: {', '.join(AVAILABLE_MODELS)}")
        return None
    
    # Set the model in client registry
    client_registry.set_primary(model_name)
    print(f"Testing with model: {model_name}")
    if use_semantic_enrichment:
        print("Semantic enrichment: ENABLED")

    print(f"Loading test data from: {TEST_DATA_FILE}")
    try:
        with open(TEST_DATA_FILE, "r") as f:
            command_dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Test data file not found at {TEST_DATA_FILE}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {TEST_DATA_FILE}")
        return None

    test_cases = [
        (command["string_cmd"], command["structured_cmd"], command.get("cmd_category", "Unknown"))
        for command in command_dataset[STARTING_CASE:]  # Adjust slice as needed
    ]

    print(f"Loaded {len(test_cases)} test cases.")

    # Collect all test results
    all_test_results = []
    execution_times = []

    for i, (input_str, expected_str, cmd_category) in enumerate(tqdm(test_cases, desc="Running BAML tests")):
        time.sleep(3)
        print(f"\n--- Test Case {STARTING_CASE + i} ---")
        print(f"Original Input: {input_str}")

        # Apply semantic enrichment if flag is active
        processed_input = input_str
        if use_semantic_enrichment:
            try:
                print("Applying semantic enrichment...")
                enriched_command = b.GenerateSemanticEnrichedCommand(input_str)
                processed_input = enriched_command
                print(f"Enriched Input: {processed_input}")
            except Exception as e:
                print(f"Warning: Semantic enrichment failed, using original input. Error: {e}")
                processed_input = input_str
        
        print(f"Processing Input: {processed_input}")

        expected_command_list = parse_expected_output(expected_str)
        if not expected_command_list:
            print(" \x1b[91mFailed (Skipping due to parse error in expected output)\x1b[0m")
            test_result = TestResult(
                index=STARTING_CASE + i,
                input_command=processed_input,
                expected_command_count=0,
                actual_command_count=0,
                score=0.0,
                passed=False,
                execution_time=0.0,
                error="Failed to parse expected output JSON",
                cmd_category=cmd_category
            )
            all_test_results.append(test_result)
            continue

        try:
            start_time = time.time()
            # Call the BAML function with the selected model using processed input
            actual_command_list = execute_command_with_model(processed_input, model_name)
            end_time = time.time()
            duration = end_time - start_time
            execution_times.append(duration)
            
            print(f"Expected: {expected_command_list.model_dump_json(indent=2)}")
            print(f"BAML Response ({duration:.2f}s): {actual_command_list.model_dump_json(indent=2)}")

            # Compare results
            score = compare_commands(actual_command_list, expected_command_list)
            print(f"Comparison Score: {score:.3f}")
            
            passed = score >= OVERALL_THRESHOLD
            if passed:
                print(f" \x1b[92mPassed (Score: {score:.3f} >= {OVERALL_THRESHOLD})\x1b[0m")
            else:
                print(f" \x1b[91mFailed (Score: {score:.3f} < {OVERALL_THRESHOLD})\x1b[0m")

            # Create test result
            test_result = TestResult(
                index=STARTING_CASE + i,
                input_command=processed_input,
                expected_command_count=len(expected_command_list.commands),
                actual_command_count=len(actual_command_list.commands),
                score=score,
                passed=passed,
                execution_time=duration,
                expected_commands=[cmd.model_dump() for cmd in expected_command_list.commands],
                actual_commands=[cmd.model_dump() for cmd in actual_command_list.commands],
                cmd_category=cmd_category
            )
            all_test_results.append(test_result)

        except Exception as e:
            print(f" \x1b[91mFailed (Error during BAML call or comparison): {e}\x1b[0m")
            duration = time.time() - start_time if "start_time" in locals() else 0.0
            execution_times.append(duration)
            
            test_result = TestResult(
                index=STARTING_CASE + i,
                input_command=processed_input,
                expected_command_count=len(expected_command_list.commands) if expected_command_list else 0,
                actual_command_count=0,
                score=0.0,
                passed=False,
                execution_time=duration,
                error=str(e),
                cmd_category=cmd_category
            )
            all_test_results.append(test_result)

    # Group results by command count
    groups_by_count = defaultdict(list)
    for result in all_test_results:
        groups_by_count[result.expected_command_count].append(result)

    # Create command count groups
    command_count_groups = []
    for command_count, results in groups_by_count.items():
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        pass_rate = (passed_count / len(results)) * 100 if results else 0
        
        group = CommandCountGroup(
            command_count=command_count,
            passed_count=passed_count,
            failed_count=failed_count,
            total_count=len(results),
            pass_rate=pass_rate,
            test_results=results
        )
        command_count_groups.append(group)

    # Group results by task type (handle multi-category assignments like A|B|C and A|C)
    task_type_results = defaultdict(list)
    
    for result in all_test_results:
        cmd_category = result.cmd_category
        if cmd_category and cmd_category != "Unknown":
            # Split multi-category assignments (e.g., "A|B|C" -> ["A", "B", "C"])
            task_types = cmd_category.split('|')
            for task_type in task_types:
                task_type = task_type.strip()
                if task_type in TASK_TYPE_DESCRIPTIONS:
                    task_type_results[task_type].append(result)
                else:
                    print(f"Warning: Unknown task type '{task_type}' in category '{cmd_category}' for command '{result.input_command}'")
        else:
            print(f"Warning: No valid task category found for command '{result.input_command}'")

    # Create task type groups
    task_type_groups = []
    for task_type, results in task_type_results.items():
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count
        pass_rate = (passed_count / len(results)) * 100 if results else 0
        
        group = TaskTypeGroup(
            task_type=task_type,
            task_description=TASK_TYPE_DESCRIPTIONS[task_type],
            passed_count=passed_count,
            failed_count=failed_count,
            total_count=len(results),
            pass_rate=pass_rate,
            test_results=results
        )
        task_type_groups.append(group)

    # Calculate overall statistics
    total_passed = sum(1 for r in all_test_results if r.passed)
    total_failed = len(all_test_results) - total_passed
    overall_pass_rate = (total_passed / len(all_test_results)) * 100 if all_test_results else 0
    average_time = np.mean(execution_times) if execution_times else 0

    # Create summary
    summary = TestSummary(
        model_name=model_name,
        total_cases=len(all_test_results),
        total_passed=total_passed,
        total_failed=total_failed,
        overall_pass_rate=overall_pass_rate,
        average_execution_time=average_time,
        groups_by_command_count=command_count_groups,
        groups_by_task_type=task_type_groups
    )

    # Print summary
    summary.print_summary()
    
    return summary


def display_available_models():
    """Display available models for selection"""
    print("\n=== Available Models ===")
    for i, model in enumerate(AVAILABLE_MODELS, 1):
        print(f"{i}. {model}")
    print("========================\n")


def select_model_interactive():
    """Interactive model selection"""
    display_available_models()
    
    while True:
        try:
            choice = input(f"Enter model number (1-{len(AVAILABLE_MODELS)}) or model name [default: {DEFAULT_MODEL}]: ").strip()
            
            if not choice:
                return DEFAULT_MODEL
            
            # Try to parse as number
            if choice.isdigit():
                choice_num = int(choice) - 1
                if 0 <= choice_num < len(AVAILABLE_MODELS):
                    return AVAILABLE_MODELS[choice_num]
                else:
                    print(f"Invalid choice. Please select a number between 1 and {len(AVAILABLE_MODELS)}")
                    continue
            
            # Try to find exact model name match
            if choice in AVAILABLE_MODELS:
                return choice
            
            # Try case-insensitive match
            for model in AVAILABLE_MODELS:
                if model.lower() == choice.lower():
                    return model
            
            print(f"Model '{choice}' not found. Available models:")
            for model in AVAILABLE_MODELS:
                print(f"  - {model}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            return None
        except Exception as e:
            print(f"Error: {e}")


def select_semantic_enrichment():
    """Interactive semantic enrichment selection"""
    while True:
        try:
            choice = input("Enable semantic enrichment? (y/N): ").strip().lower()
            if choice in ['', 'n', 'no']:
                return False
            elif choice in ['y', 'yes']:
                return True
            else:
                print("Please enter 'y' for yes or 'n' for no.")
        except KeyboardInterrupt:
            print("\nExiting...")
            return None
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Interactive model selection
    selected_model = select_model_interactive()
    
    if selected_model:
        # Interactive semantic enrichment selection
        use_enrichment = select_semantic_enrichment()
        
        if use_enrichment is not None:
            enrichment_suffix = "_enriched" if use_enrichment else ""
            print(f"\nStarting tests with model: {selected_model}")
            if use_enrichment:
                print("Semantic enrichment will be applied to input commands.")
            
            results = run_tests(selected_model, use_enrichment)
            
            if results:
                print(f"\nTest execution completed. Results available in structured format.")
                print(f"Use the returned TestSummary object to access detailed results.")
                
                # Example: Save results to JSON file
                results_file = f"test_results_{selected_model}{enrichment_suffix}_{int(time.time())}.json"
                with open(results_file, 'w') as f:
                    json.dump(results.model_dump(), f, indent=2)
                print(f"Results saved to: {results_file}")
            else:
                print("Test execution failed.")
        else:
            print("No semantic enrichment option selected. Exiting.")
    else:
        print("No model selected. Exiting.")
