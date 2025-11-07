import sys
import os
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import secrets

# Add the parent directory to Python path to enable absolute imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baml_client.sync_client import b
from baml_py import ClientRegistry
from dotenv import load_dotenv
from command_interpreter.caller import execute_function, clear_command_history
from command_interpreter.tasks import Tasks

# Import command generator setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dataset_generator'))
from gpsr_commands_structured import CommandGenerator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'dataset_generator', 'CommandGenerator', 'src'))
try:
    from robocupathome_generator.generator import (
        read_data, parse_names, parse_locations, parse_rooms, parse_objects
    )
    GENERATOR_FUNCTIONS_AVAILABLE = True
except ImportError:
    GENERATOR_FUNCTIONS_AVAILABLE = False

# Load environment variables
load_dotenv()

# Set log level if available
try:
    from baml_client.config import set_log_level
    set_log_level("ERROR")
except ImportError:
    pass  # Config module not available, continue without it

# Initialize FastAPI app
app = FastAPI(title="FRIDA Command Interpreter API", version="1.0.0")

# Security: API Key validation
API_KEY = os.getenv("BACKEND_API_KEY")
if not API_KEY:
    # Generate a random API key if not set (for development only)
    API_KEY = secrets.token_urlsafe(32)
    print(f"WARNING: No BACKEND_API_KEY set. Generated temporary key: {API_KEY}")

# CORS configuration - restrict to your Vercel frontend domain in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Available models (excluding local models for web deployment)
AVAILABLE_MODELS = [
    'GEMINI_PRO_2_5',
    'GEMINI_FLASH_2_5',
    'OPENAI_GPT_4_1_MINI',
    'ANTHROPIC_CLAUDE_SONNET_4',
    'META_LLAMA_3_3_8B_IT_FREE',
    'META_LLAMA_3_3_70B'
]

# Initialize client registry
client_registry = ClientRegistry()

# Create a single Tasks instance
tasks = Tasks()

# Setup command generator
command_generator = None

def setup_command_generator():
    """Set up the CommandGenerator with data from CompetitionTemplate"""
    try:
        base_path = os.path.join(os.path.dirname(__file__), '..', 'dataset_generator')
        names_file_path = os.path.join(base_path, 'CompetitionTemplate', 'names', 'names.md')
        locations_file_path = os.path.join(base_path, 'CompetitionTemplate', 'maps', 'location_names.md')
        rooms_file_path = os.path.join(base_path, 'CompetitionTemplate', 'maps', 'room_names.md')
        objects_file_path = os.path.join(base_path, 'CompetitionTemplate', 'objects', 'objects.md')

        if GENERATOR_FUNCTIONS_AVAILABLE:
            names_data = read_data(names_file_path)
            names = parse_names(names_data)

            locations_data = read_data(locations_file_path)
            location_names, placement_location_names = parse_locations(locations_data)

            rooms_data = read_data(rooms_file_path)
            room_names = parse_rooms(rooms_data)

            objects_data = read_data(objects_file_path)
            object_names, object_categories_plural, object_categories_singular = parse_objects(objects_data)

            return CommandGenerator(names, location_names, placement_location_names, room_names, object_names,
                                   object_categories_plural, object_categories_singular)
        return None
    except Exception as e:
        print(f"Warning: Could not set up CommandGenerator: {e}")
        return None

# Initialize command generator at startup
@app.on_event("startup")
async def startup_event():
    global command_generator
    command_generator = setup_command_generator()

# Security: API Key dependency
async def verify_api_key(authorization: Optional[str] = Header(None)):
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header required")

    # Expected format: "Bearer <API_KEY>"
    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authorization format")

    if parts[1] != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True

# Pydantic models
class InterpretRequest(BaseModel):
    command: str
    model: str
    execute: bool = False  # Whether to actually execute commands or just parse

class GenerateCommandRequest(BaseModel):
    model: str
    execute: bool = False

class CommandResponse(BaseModel):
    commands: list
    string_command: Optional[str] = None
    execution_results: Optional[list] = None

# Routes
@app.get("/")
async def root():
    return {
        "message": "FRIDA Command Interpreter API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/models")
async def get_models(authorized: bool = Depends(verify_api_key)):
    """Get list of available models"""
    return {"models": AVAILABLE_MODELS}

@app.post("/interpret", response_model=CommandResponse)
async def interpret_command(
    request: InterpretRequest,
    authorized: bool = Depends(verify_api_key)
):
    """Interpret a natural language command"""
    if not request.command or not request.model:
        raise HTTPException(status_code=400, detail="Missing command or model")

    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model")

    try:
        # Set the model
        client_registry.set_primary(request.model)

        # Generate command list using BAML
        command_list = b.GenerateCommandList(
            request.command,
            baml_options={"client_registry": client_registry}
        )

        # Convert commands to dictionaries
        commands = []
        execution_results = []

        if hasattr(command_list, 'commands') and command_list.commands:
            for cmd in command_list.commands:
                if hasattr(cmd, 'model_dump'):
                    cmd_dict = cmd.model_dump()
                    commands.append(cmd_dict)

                    # Execute if requested
                    if request.execute:
                        try:
                            action, success, result = execute_function(cmd, tasks, grounding=True)
                            execution_results.append({
                                "action": action,
                                "success": success,
                                "result": result
                            })
                        except Exception as e:
                            execution_results.append({
                                "action": str(cmd),
                                "success": False,
                                "result": str(e)
                            })

        # Clear command history after execution
        if request.execute:
            clear_command_history(tasks)

        return CommandResponse(
            commands=commands,
            string_command=request.command,
            execution_results=execution_results if request.execute else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate", response_model=CommandResponse)
async def generate_command(
    request: GenerateCommandRequest,
    authorized: bool = Depends(verify_api_key)
):
    """Generate a random command and optionally execute it"""
    if not command_generator:
        raise HTTPException(
            status_code=503,
            detail="Command generator is not available. Check CompetitionTemplate setup."
        )

    if request.model not in AVAILABLE_MODELS:
        raise HTTPException(status_code=400, detail="Invalid model")

    try:
        # Generate a random command
        string_cmd, structured_cmd = command_generator.generate_full_command()

        # Set the model
        client_registry.set_primary(request.model)

        # Generate command list using BAML
        command_list = b.GenerateCommandList(
            string_cmd,
            baml_options={"client_registry": client_registry}
        )

        # Convert commands to dictionaries
        commands = []
        execution_results = []

        if hasattr(command_list, 'commands') and command_list.commands:
            for cmd in command_list.commands:
                if hasattr(cmd, 'model_dump'):
                    cmd_dict = cmd.model_dump()
                    commands.append(cmd_dict)

                    # Execute if requested
                    if request.execute:
                        try:
                            action, success, result = execute_function(cmd, tasks, grounding=True)
                            execution_results.append({
                                "action": action,
                                "success": success,
                                "result": result
                            })
                        except Exception as e:
                            execution_results.append({
                                "action": str(cmd),
                                "success": False,
                                "result": str(e)
                            })

        # Clear command history after execution
        if request.execute:
            clear_command_history(tasks)

        return CommandResponse(
            commands=commands,
            string_command=string_cmd,
            execution_results=execution_results if request.execute else None
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
