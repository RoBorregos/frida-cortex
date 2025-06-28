"""Testing three different approaches to achieve structured outputs:
1. BAML (Schema Aligned Parsing)
2. JSON mode
3. Prompting
"""

import json
import time
from typing import Optional, List, Dict
from collections import defaultdict

# Updated imports to reference parent directory level
import sys
import os
from dotenv import load_dotenv
from baml_py import ClientRegistry, Collector
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
from tqdm import tqdm
import numpy as np
from pydantic import BaseModel

load_dotenv()
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from baml_client.sync_client import b
from baml_client.types import CommandListLLM
from baml_client.config import set_log_level

# Turn off all logging
set_log_level("ERROR")

DEFAULT_MODEL = "GEMINI_FLASH_2_5"

# Initialize client registry
client_registry = ClientRegistry()
collector = Collector()

# --- Configuration ---
STARTING_CASE = 115  # Adjust if needed
SIMILARITY_THRESHOLD = 0.8  # Threshold for complement similarity
OVERALL_THRESHOLD = 0.75  # Threshold for the overall test case score
TEST_DATA_FILE = "../../dataset_generator/dataset.json"
TEST_DATA_FILE_ENRICHED_AND_REORDERED = "../../dataset_generator/dataset_enriched_reordered.json"
BAML_ENABLED = True

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Example model

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
    input_tokens: int
    output_tokens: int
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
    average_input_tokens: int
    average_output_tokens: int
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
        print(f"Average Input Tokens: {self.average_input_tokens}")
        print(f"Average Output Tokens: {self.average_output_tokens}")
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

def execute_command_with_model(command_text: str, mode: int):
    """Execute a command using the appropriate model function"""
    if mode == 1:
        return b.GenerateCommandList(command_text,
                                     baml_options={"client_registry": client_registry,
                                                   "collector": collector})
    else:
        from google import genai

        system_prompt = """
        system:
You are a service robot for domestic applications.
 Now you are located in a house environment and we will give you general purpose tasks in the form of natural language.

 You have in your architecture the modules of:
 - navigation
 - manipulation
 - person recognition
 - object detection
 - human-robot interaction

 Your job is to understand the task and divide it into smaller actions proper to your modules,
 considering a logical flow of the actions along the time. For example, for the prompt
 'Locate a dish in the kitchen then get it and give it to Angel in the living room', the result would be: 

 {
     "commands": [
         {'action': 'go_to', 'location_to_go': 'kitchen'},
         {'action': 'pick_object', 'object_to_pick': 'dish'},
         {'action': 'go_to', 'location_to_go': 'living room'},
         {'action': 'find_person_by_name', 'name': 'Angel'},
         {'action': 'give_object'}
     ]
 }

 Another important thing is that when we give you the general task, some complements are grouped in categories.
 For example: apple, banana and lemon are all of them in the fruits category; cola, milk and red wine are all of them in the drinks category.
 If we give you a task talking about an item category, do not change the category word. It is very important you don't make up information
 not given explicitly. If you add new words, we will be disqualified. 
 For example:  'navigate to the bedroom then locate a food'.

 Another important thing is that you have to rememeber the name of the person, in case we are talking about 
 someone specifically. An example for the prompt can be: 'Get a snack from the 
 side tables and deliver it to Adel in the bedroom'.

 The system will handle the instructions and collect the response for each command.
 You can set the commands to use information from previous commands and context, for example:
 'tell me how many foods there are on the bookshelf'
 {
  "commands": [
    {"action": "go_to", "location_to_go": "bookshelf"},
    {"action": "count", "target_to_count": "foods"},
    {"action": "go_to", "location_to_go": "start_location"},
    {"action": "say_with_context", "user_instruction": "how many foods there are on the bookshelf", "previous_command_info": "count"}
  ]
 }

Answer in JSON using this schema:
{
  // List of commands for the robot to execute
  commands: [
    {
      action: "go_to",
      // The location to go to,
      //   (kitchen, living room table, etc.)
      //   if asked to return something to the initial user use 'start_location'
      location_to_go: "start_location" or string,
    } or {
      action: "pick_object",
      // Name of the object to pick
      object_to_pick: string,
    } or {
      action: "find_person_by_name",
      // Name of the person to find
      name: string,
    } or {
      action: "find_person",
      // Can be a feature/pose (pointing to
      //   the left, standing, blue shirt, etc.). If just asked to find ANY person, leave empty
      attribute_value: "" or string,
    } or {
      action: "count",
      // Name of the object/person to count
      //   (snacks, persons pointing to the left, persons wearing blue shirt, etc.)
      target_to_count: string,
    } or {
      action: "get_person_info",
      // Information to be
      //   retrieved about the person (pose, gesture, name, clothing, age, etc.)
      info_type: "pose" or "gesture" or "name" or string,
    } or {
      action: "get_visual_info",
      // The property which will be measured to find the
      //   desired object in the environment (color, shape, size, thinnest, biggest, etc.)
      measure: string,
      // The category of the object to be found
      //   (snack, drink, dish, object, etc.)
      object_category: string,
    } or {
      action: "answer_question",
    } or {
      action: "follow_person_until",
      // The destination location to which
      //   the robot will follow the person until, can be a room, furniture, etc. or 
      //   'cancelled' if the robot is to follow the person until asked to stop
      destination: "cancelled" or string,
    } or {
      action: "guide_person_to",
      // The destination to guide or lead a person to
      //   can be a room, furniture, etc.
      destination_room: string,
    } or {
      action: "give_object",
    } or {
      action: "place_object",
    } or {
      action: "say_with_context",
      // Instruction that will help to system
      //   to return the desired information to the user, it can be a question or a
      //   statement like 'tell me how many foods there are on the bookshelf'
      user_instruction: string,
      // Previous command needed to return
      //   the desired information to the user, it has to be a previous command that
      //   have been executed ('count', 'get_visual_info', etc.), or
      //   internal information that the system has stored in its memory 
      //   ('time', 'affection', etc.)
      previous_command_info: "introduction" or string,
    }
  ],
}

user:
Generate commands for this prompt in the specified format.
        """
        config = {}
        if mode == 2:
            config={
                "response_mime_type": "application/json",
            }
    
        client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=system_prompt + command_text,
            config=config
        )

        my_commands: CommandListLLM = CommandListLLM(**json.loads(response.text))
        return my_commands


def run_tests(model_name: str = DEFAULT_MODEL, mode: int = 1) -> TestSummary:
    """Loads data, runs tests, and returns structured results.
    
    Args:
        model_name: The model to use for testing. Must be one of AVAILABLE_MODELS.
        use_enrichment_and_reorder: If True, applies combined enrichment and reordering to input commands before processing.
        
    Returns:
        TestSummary: Structured results including grouping by command count
    """
    
    # Set the model in client registry
    client_registry.set_primary(model_name)
    print(f"Testing with model: {model_name}")
    test_data_file = TEST_DATA_FILE

    print(f"Loading test data from: {test_data_file}")
    try:
        with open(test_data_file, "r") as f:
            command_dataset = json.load(f)
    except FileNotFoundError:
        print(f"Error: Test data file not found at {test_data_file}")
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
    total_input_tokens = []
    total_output_tokens = []
    for i, (input_str, expected_str, cmd_category) in enumerate(tqdm(test_cases, desc="Running BAML tests")):
        time.sleep(2)
        print(f"\n--- Test Case {STARTING_CASE + i} ---")
        print(f"Input: {input_str}")

        expected_command_list = parse_expected_output(expected_str)
        if not expected_command_list:
            print(" \x1b[91mFailed (Skipping due to parse error in expected output)\x1b[0m")
            test_result = TestResult(
                index=STARTING_CASE + i,
                input_command=input_str,
                expected_command_count=0,
                actual_command_count=0,
                score=0.0,
                passed=False,
                execution_time=0.0,
                input_tokens=0,
                output_tokens=0,
                error="Failed to parse expected output JSON",
                cmd_category=cmd_category
            )
            all_test_results.append(test_result)
            continue

        try:
            start_time = time.time()
            # Call the BAML function with the selected model using processed input
            actual_command_list = execute_command_with_model(input_str, mode)
            end_time = time.time()
            duration = end_time - start_time
            execution_times.append(duration)
            input_tokens = 0
            output_tokens = 0
            if collector.last and collector.last.usage:
                input_tokens = collector.last.usage.input_tokens or 0
                output_tokens = collector.last.usage.output_tokens or 0
                total_input_tokens.append(input_tokens)
                total_output_tokens.append(output_tokens)
            
            print(f"Expected: {expected_command_list.model_dump_json(indent=2)}")
            print(f"BAML Response ({duration:.2f}s): {actual_command_list.model_dump_json(indent=2)}")

            # Compare results
            score = compare_commands(actual_command_list, expected_command_list)
            # Override with LLM GetFeasibilityScore
            #feasibility_score = b.GetFeasibilityScore(input_str, expected_command_list, actual_command_list)
            #score = feasibility_score.score
            print(f"Comparison Score: {score:.3f}")
            
            passed = score >= OVERALL_THRESHOLD
            if passed:
                print(f" \x1b[92mPassed (Score: {score:.3f} >= {OVERALL_THRESHOLD})\x1b[0m")
            else:
                print(f" \x1b[91mFailed (Score: {score:.3f} < {OVERALL_THRESHOLD})\x1b[0m")

            # Create test result
            test_result = TestResult(
                index=STARTING_CASE + i,
                input_command=input_str,
                expected_command_count=len(expected_command_list.commands),
                actual_command_count=len(actual_command_list.commands),
                score=score,
                passed=passed,
                execution_time=duration,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
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
                input_command=input_str,
                expected_command_count=len(expected_command_list.commands) if expected_command_list else 0,
                actual_command_count=0,
                score=0.0,
                passed=False,
                execution_time=duration,
                input_tokens=0,
                output_tokens=0,
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
    average_input_tokens = int(np.mean(total_input_tokens)) if total_input_tokens else 0
    average_output_tokens = int(np.mean(total_output_tokens)) if total_output_tokens else 0
    # Create summary
    summary = TestSummary(
        model_name=model_name,
        total_cases=len(all_test_results),
        total_passed=total_passed,
        total_failed=total_failed,
        overall_pass_rate=overall_pass_rate,
        average_execution_time=average_time,
        average_input_tokens=average_input_tokens,
        average_output_tokens=average_output_tokens,
        groups_by_command_count=command_count_groups,
        groups_by_task_type=task_type_groups
    )

    # Print summary
    summary.print_summary()
    
    return summary


def display_available_modes():
    """Display available modes for structured outputs"""
    print("\n=== Available Modes ===")
    print("1. BAML (Schema Aligned Parsing)")
    print("2. JSON mode")
    print("3. Prompting")
    print("0. Exit")
    print("========================\n")


def select_mode_interactive():
    """Interactive mode selection"""
    display_available_modes()
    
    while True:
        try:
            choice = int(input("Enter mode number (1-3): "))
            
            if not choice:
                return 0
            
            return choice
            
        except KeyboardInterrupt:
            print("\nExiting...")
            return None
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Interactive model selection
    selected_mode = select_mode_interactive()
    
    if selected_mode > 0:
        print(f"\nStarting tests with mode: {selected_mode}")

        results = run_tests(DEFAULT_MODEL, selected_mode)
            
        if results:
            print(f"\nTest execution completed. Results available in structured format.")
            print(f"Use the returned TestSummary object to access detailed results.")
            
            # Example: Save results to JSON file inside results folder
            selected_mode_suffix = "baml" if selected_mode == 1 else "json_mode" if selected_mode == 2 else "prompting"
            results_file = f"results/structured_results_{selected_mode_suffix}_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(results.model_dump(), f, indent=2)
            print(f"Results saved to: {results_file}")
        else:
            print("Test execution failed.")
    else:
        print("No mode selected. Exiting.")
