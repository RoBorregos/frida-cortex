import pandas as pd
import json
import re

def classify_command(command_text):
    """
    Classifies a command into one of three categories: A, B, or C.
    A: Navigation and following people.
    B: Object manipulation.
    C: Speaking or answering questions.
    """
    command_text = command_text.lower()
    
    # Category A: High-priority keywords for following/guiding
    if any(k in command_text for k in ['follow', 'guide', 'accompany', 'lead', 'escort']):
        return 'A'

    # Category C: Keywords for speaking/answering. These often define the main goal.
    if any(k in command_text for k in ['tell', 'say', 'greet', 'introduce', 'ask', 'answer', "what's", 'how many', 'pose of', 'name of', 'gender of', 'joke', 'question']):
        return 'C'

    # Category B: Keywords for object interaction.
    if any(k in command_text for k in ['bring', 'take', 'get', 'put', 'place', 'deliver', 'give', 'serve', 'distribute', 'arrange', 'grasp', 'pick up', 'dump', 'take out']):
        return 'B'

    # More ambiguous keywords: find, locate, look for
    if any(k in command_text for k in ['find', 'locate', 'look for']):
        # A list of common objects and categories found in the commands
        object_keywords = [
            'orange', 'cloth', 'bag', 'paprika', 'tray', 'snacks', 'bowl', 'valise', 'cup', 'fruits', 
            'cutlery', 'potato chips', 'cereal', 'coke', 'sausages', 'fork', 'chocolate drink', 
            'pringles', 'cascade pod', 'sponge', 'crackers', 'knife', 'grape juice', 'food', 
            'tableware', 'drinks', 'cleaning stuff', 'object', 'containers', 'apple', 'dish', 'spoon'
        ]
        
        if any(obj in command_text for obj in object_keywords):
            return 'B'
        
        # Check for person-related terms if no object is found
        person_keywords = [
            'person', 'people', 'guest', 'man', 'woman', 'boy', 'girl', 
            'children', 'elders', 'waving', 'sitting', 'standing', 'pointing', 'raising'
        ]
        if any(p in command_text for p in person_keywords):
            return 'C'

    # Simple navigation commands
    if 'go to' in command_text or 'navigate to' in command_text or re.match(r"^\s*the \w+\s*$", command_text):
        return 'A'

    # Default to 'B' for commands that might imply object interaction without explicit keywords
    # This acts as a catch-all for commands that are not clearly A or C.
    return 'B'


def create_legacy_dataset():
    """
    Reads commands from a CSV file, classifies them, and saves them to a JSON file.
    """
    try:
        df = pd.read_csv('generated_commands.csv')
    except FileNotFoundError:
        print("Error: generated_commands.csv not found. Make sure the file is in the 'legacy_dataset_generator' directory.")
        return

    commands = df['Command'].head(100)
    
    dataset = []
    for cmd in commands:
        category = classify_command(cmd)
        dataset.append({
            "string_cmd": cmd,
            "cmd_category": category,
            "structured_command": None,
            "cmd_type": None
        })
        
    # Save the dataset to a file
    with open('dataset_legacy.json', 'w') as f:
        json.dump(dataset, f, indent=4)
        
    print("Dataset with 100 commands created successfully in 'dataset.json'")
    
    # Print distribution
    categories = [item['cmd_category'] for item in dataset]
    distribution = {cat: categories.count(cat) for cat in set(categories)}
    print("Command distribution:", distribution)


if __name__ == "__main__":
    create_legacy_dataset() 