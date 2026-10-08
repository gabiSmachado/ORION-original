import httpx, yaml, sys, os
from fastmcp import FastMCP
from utils.logger import get_logger
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from typing import Any, Dict
import json
from utils.models import CreateSession



logger = get_logger("mcp-server", log_file="mcp_server.log", level=20, console_level=20)

CONFIG_FILE_PATH = 'config/config.yaml'

mcp = FastMCP("nsb-mcp")

@mcp.tool()
async def create_session(body: CreateSession) -> Dict[str, Any]:
    """Create a new Network Slice Booking session.

    Provide a JSON body matching the API's CreateSession schema.
    Returns the API response. If the API wraps the created SessionInfo in a
    stringified 'message' field, this tool will parse that JSON for convenience
    under messageParsed.
    """
    url = f"{API_ROOT}/sessions"
    body = body.model_dump(mode="json")
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            logger.info(f"create_session -> POST {url} payload={body}")
            resp = await client.post(url, json=body)
    except Exception as e:
        logger.error(f"create_session connect error: {e}")
        return {"status": 503, "code": "CONNECTION_ERROR", "message": str(e)}

    try:
        result: Dict[str, Any]
        result = resp.json()
    except Exception:
        return {"status": resp.status_code, "code": "HTTP_ERROR", "message": resp.text}
    
    # Normalize potential stringified JSON in the 'message' field from the API
    if isinstance(result, dict) and "message" in result and isinstance(result["message"], str):
        try:
            result["messageParsed"] = json.loads(result["message"])
        except Exception:
            # leave as-is if not JSON
            pass
    return result


@mcp.tool()
async def get_session(session_id: str) -> Dict[str, Any]:
    """Get information about a session by its UUID string."""
    url = f"{API_ROOT}/sessions/{session_id}"
    logger.info(f"get_session -> GET {url}")
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url)
    try:
        return resp.json()
    except Exception:
        return {"status": resp.status_code, "code": "HTTP_ERROR", "message": resp.text}


@mcp.tool()
async def delete_session(session_id: str) -> Dict[str, Any]:
    """Delete a session by its UUID string."""
    url = f"{API_ROOT}/sessions/{session_id}"
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.delete(url)
    try:
        return resp.json()
    except Exception:
        return {"status": resp.status_code, "code": "HTTP_ERROR", "message": resp.text}


@mcp.tool()
async def ping_api() -> Dict[str, Any]:
    """Quick health-check: GET the API root path to verify connectivity."""
    # The API doesn't expose a root health endpoint; try a harmless GET to a non-existing resource
    # and report reachability based on HTTP status.
    test_url = f"{API_ROOT}/__health__"
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(test_url)
        return {"reachable": True, "status": resp.status_code, "url": test_url}
    except Exception as e:
        return {"reachable": False, "error": str(e), "url": test_url}
    
if __name__ == "__main__":    
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            raw_cfg = yaml.safe_load(f) or {}
            # Accept either flat host/port or nested under mcp_server
            section = raw_cfg.get('mcp_server', raw_cfg)
            config = {
                'host': os.getenv('MCP_HOST', section.get('host', '0.0.0.0')),
                'port': int(os.getenv('MCP_PORT', section.get('port', 8000)))
            }
            api = raw_cfg.get('slice_api', raw_cfg)
            API_ROOT = f"http://{api.get('host')}:{api.get('port')} "
            client = httpx.AsyncClient(base_url= API_ROOT)
    except FileNotFoundError:
        print(f"Configuration file not found: {CONFIG_FILE_PATH}")
        sys.exit(1)
    except yaml.YAMLError as exc:
        print(f"Error parsing configuration file: {exc}")
        sys.exit(1)

    logger.info(f"Slice API: {API_ROOT})")
    logger.info(f"MCP server binding {config.get('host')}:{config.get('port')}")
    
    try:
        logger.info("Starting MCP server...")
        # Inject Starlette middleware to log incoming HTTP requests
        mcp.run(
            transport="sse",
            host=config.get('host'),
            port=config.get('port')
        )
        logger.info("API")
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Failed to start MCP server: {e}")
        sys.exit(1)