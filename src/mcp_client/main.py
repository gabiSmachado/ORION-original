import sys
import os
import yaml
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from client import MCPClient
from dotenv import load_dotenv
from utils.logger import get_logger
from pathlib import Path

logger = get_logger("mcp-client", log_file="mcp_client.log", level=20, console_level=20)

load_dotenv()
CONFIG_FILE_PATH = 'config/config.yaml'

try:
    logger.info(f"Starting MCP Client API.")
    
    with open(CONFIG_FILE_PATH, 'r') as f:
        raw_cfg = yaml.safe_load(f) or {}
        config_server =  raw_cfg.get('mcp_server', raw_cfg)
        server_url = f"http://{config_server.get('host')}:{config_server.get('port')}/sse"
        
        config_client = raw_cfg.get('mcp_client', raw_cfg)
        client_host = config_client.get('host')
        client_port = config_client.get('port')
        
        rapp_config = raw_cfg.get('rapp', raw_cfg)
        rapp_url = f"http://{rapp_config.get('host')}:{rapp_config.get('port')}"

        llm_config = raw_cfg.get('llm', raw_cfg)
        llm = llm_config.get('name')
        model = llm_config.get('model')
        results_path = raw_cfg.get('results_path', raw_cfg)
        file_path = Path(f"{results_path}/{llm}.csv")
except FileNotFoundError:
    print(f"Configuration file not found: {CONFIG_FILE_PATH}")
    logger.info(f"Configuration file not found: {CONFIG_FILE_PATH}")
    sys.exit(1)
except yaml.YAMLError as exc:
    print(f"Error parsing configuration file: {exc}")
    sys.exit(1)

try:  
    if llm == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
    elif llm == "anthropic":
        api_key = os.getenv("ANTHROPIC_API_KEY")
    elif llm == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
    elif llm == "qwen":
        api_key = os.getenv("OPENROUTER_API_KEY")
except EnvironmentError:
    print("Not found an API_KEY in your environment or .env")
    sys.exit(1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    client = MCPClient(logger=logger, rapp=rapp_url, file_path=file_path)
    try:
        connected = await client.connect_to_server(server_url)
        if not connected:
            raise Exception("Failed to connect to server")
        app.state.client = client
        yield
    except Exception as e:
        raise Exception(f"Failed to connect to server: {str(e)}")
    finally:
        # Shutdown
        await client.cleanup()

app = FastAPI(title="MCP Chatbot API", lifespan=lifespan)

class IntentRequest(BaseModel):
    intent: str

@app.post("/intent")
async def process_query(request: IntentRequest):
    """Process a user intent and return updated conversation messages."""
    logger.info(f"Processing intent payload: {request.intent[:120]}...")
    try:
        await app.state.client.set_llm(llm_name=llm,llm_model=model, api_key=api_key)
        messages = await app.state.client.process_intent(request.intent)
        return messages
    except Exception as e:
        logger.error(f"/intent failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host=client_host, port=client_port)