import sys
import os
from flask import Flask, render_template, request, jsonify

# Add the parent directory to Python path to enable absolute imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baml_client.sync_client import b
from baml_py import ClientRegistry
from dotenv import load_dotenv
from command_interpreter.caller import execute_function, clear_command_history
from command_interpreter.tasks import Tasks
from termcolor import colored

# Load environment variables from .env file
load_dotenv()

# Available models from client.baml
AVAILABLE_MODELS = [
    'LOCAL_FINETUNED',
    'GEMINI_PRO_2_5',
    'GEMINI_FLASH_2_5',
    'OPENAI_GPT_4_1_MINI',
    'ANTHROPIC_CLAUDE_SONNET_4',
    'META_LLAMA_3_3_8B_IT_FREE',
    'META_LLAMA_3_3_70B'
]

# Initialize Flask app
app = Flask(__name__)

# Initialize client registry
client_registry = ClientRegistry()

# Create a single Tasks instance to reuse throughout the session
tasks = Tasks()

def execute_command(command_text, model_name):
    """Execute a command using the appropriate model function"""
    try:
        client_registry.set_primary(model_name)

        if model_name == "LOCAL_FINETUNED":
            command_list = b.GenerateCommandListFineTuned(command_text,
                                                          baml_options={"client_registry": client_registry})
        else:
            command_list = b.GenerateCommandList(command_text,
                                                 baml_options={"client_registry": client_registry})

        results = []
        for cmd in command_list.commands:
            # Instead of executing, we'll just return the command details
            results.append(cmd.model_dump())

        clear_command_history(tasks)
        return results

    except Exception as e:
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html', models=AVAILABLE_MODELS)

@app.route('/interpret', methods=['POST'])
def interpret():
    data = request.get_json()
    command = data.get('command')
    model = data.get('model')

    if not command or not model:
        return jsonify({"error": "Missing command or model"}), 400

    if model not in AVAILABLE_MODELS:
        return jsonify({"error": "Invalid model"}), 400

    results = execute_command(command, model)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)
