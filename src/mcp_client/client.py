import asyncio
import json
from typing import Optional
from contextlib import AsyncExitStack
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types
from mcp import ClientSession
from mcp.client.sse import sse_client
import logging 
import requests
import utils.resolver as resolver
from pathlib import Path
from utils.results_file import save_results


class MCPClient:
    def __init__(self, logger: logging, rapp: str, file_path: Path):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.tools = []
        self.messages = []
        self.logger = logger
        self.rapp = rapp
        self.instructions = None
        self.llm = None
        self.llm_name = None
        self.llm_model = None
        self.result_file_path = file_path
        
    async def set_llm(self, llm_name:str,llm_model:str, api_key:str):
        try:
            self.logger.info(f"Setting llm {llm_name} - model {llm_model}")
            
            self.llm_name = llm_name
            self.llm_model = llm_model
            mcp_tools = await self.get_mcp_tools()

            initial_prompt = (
                        """
                            You are an assistant that manages network slice reservations for developers via MCP tool calls.
                            Follow these rules:
                            1. The number of devices (UEs), service time, and service area are not required values.
                            2. The number of devices does not have a defined maximum value, as it depends on the value entered by the user.
                            3. Consider the number of UEs/devices only when defined by the user.
                            4. Ask for throughput/downstream/upstream units only when the user provides a value without any unit. If the value already includes a unit (accept case-insensitive variants such as bps, kbps, Mbps, Gbps, Tbps, Mb/s, megabits per second, etc.), proceed without asking again and reuse that unit.
                            5. Service time and area stay null unless the user specifies them.
                            6. If a latency/delay budget is given without a unit, assume "Milliseconds".
                            7. Don't treat mentions of cost or budget as guidance for choosing conservative values, only populate fields that the user requested.
                            8. All fields stay null unless the user specifies them, never fabricate values.
                        """
            )
            
            if llm_name == "openai":
                self.llm =  OpenAI(api_key=api_key)       
                self.tools = [
                    {
                        "type": "function",
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema,
                    }
                    for tool in mcp_tools
                ]
            elif llm_name == "anthropic":
                self.llm =  Anthropic(api_key=api_key)
                self.tools = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema,
                    }
                    for tool in mcp_tools
                ]

            elif llm_name == "qwen":
                self.llm =  OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
                for tool in mcp_tools:              
                    self.tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": self.get_schema(tool),
                            }
                        }
                        for tool in mcp_tools
                    ]
                    
            elif llm_name == "gemini":
                self.llm =  genai.Client(api_key=api_key)
                tools_declaration = []

                for tool in mcp_tools:
                    clean_parameters = resolver.resolve_genai_schema(tool.inputSchema)
                    tools_declaration.append({
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": clean_parameters
                    })
                tools = types.Tool(function_declarations=tools_declaration)
                self.tools = types.GenerateContentConfig(tools=[tools])
                self.messages.insert(0, types.Content(
                    role="user",
                    parts=[types.Part(text=initial_prompt)],
                ))

            if llm_name != "gemini":
                self.messages.insert(0, {
                        "role": "user", 
                        "content": initial_prompt
                    }
                )
            
            self.instructions=(
                            """
                            Identify the slice type based on the user's description. It is possible to classify it into the 3 types below:
                                1. eMBB: Enhanced Mobile Broadband, focuses on delivering high data rates, strong capacity, and consistent user experience. Designed for high-bandwidth applications like streaming, gaming, and virtual reality.
                                2. uRRLC: Ultra-Reliable Low Latency Communication, supports critical applications that need a high degree of reliability and very low latency, like drone control, autonomous vehicles, industrial automation, and remote robotic surgery.
                                3. mMTC: Massive Machine Type Communications, intended to connect a vast number of devices, such as those in the Internet of Things (IoT), which requires low data rates. Used in applications where many devices are interconnected for purposes like smart sensors, connected homes, building smart cities, or monitoring environments.
                            Only output a single word in your response with no additional formatting or commentary.
                            Your response should only be one the acronym of the slice type "eMBB", "uRRLC", or "mMTC".
                            """
                        )
            self.logger.info("LLM configuration completed successfully.")
        except Exception as e:
            self.logger.error(f"Error seting LLM model: {e}")
            raise       
        
    async def connect_to_server(self, server_url: str):
        """Connect to an MCP server.

        host/port/path are passed explicitly (taken from config in main) so we don't
        silently use a hard‑coded 127.0.0.1 which breaks in Kubernetes.
        """
        self.logger.info(f"Attempting to connect to server at {server_url}.")
        try:
            result = await self.exit_stack.enter_async_context(
                sse_client(server_url)
            )
            if isinstance(result, (tuple, list)):
                if len(result) < 2:
                    raise RuntimeError("streamablehttp_client returned fewer than 2 elements; cannot get read/write streams")
                self.read_stream, self.write_stream = result[0], result[1]
            else:
                self.read_stream = getattr(result, "read_stream", getattr(result, "read", None))
                self.write_stream = getattr(result, "write_stream", getattr(result, "write", None))
                if self.read_stream is None or self.write_stream is None:
                    raise RuntimeError("Unable to locate read/write streams on streamablehttp_client result")

            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.read_stream, self.write_stream)
            )

            try:
                await self.session.initialize()
            except asyncio.CancelledError as ce:
                # Provide clearer context for the common cancellation symptom the user saw
                raise RuntimeError(
                    "Initialization cancelled. This often means the server URL/path is incorrect or the server did not respond to the MCP initialize request."
                ) from ce
            except ConnectionResetError as cre:
                raise RuntimeError(
                    "Connection was reset by the server. Ensure the transport matches (use '/sse' for SSE servers) and that the server is running and reachable."
                ) from cre

            mcp_tools = await self.get_mcp_tools()

            self.logger.info(
                f"Successfully connected to server. Available tools: {[tool.name for tool in  mcp_tools]}"
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to server at {server_url}: {e}")
            raise

    async def get_mcp_tools(self):
        try:
            self.logger.info("Requesting MCP tools from the server.")
            response = await self.session.list_tools()
            return response.tools
        except Exception as e:
            self.logger.error(f"Failed to get MCP tools: {str(e)}")
            raise Exception(f"Failed to get tools: {str(e)}")

    async def call_tool(self, tool_name: str, tool_args: dict):
        """Call a tool with the given name and arguments"""
        try:
            result = await self.session.call_tool(tool_name, tool_args)
            return result
        except Exception as e:
            self.logger.error(f"Failed to call tool: {str(e)}")
            raise Exception(f"Failed to call tool: {str(e)}")
        
    async def process_intent(self, intent: str):
        """Process an intent: prefer an LLM tool call; if none, return the LLM response.

        Returns the AI Responses API response object. If a tool is invoked, also returns
        the follow-up response after providing the tool result back to the model.
        """
        try:
            self.logger.info(f"Calling LLM: {self.llm_name}")
            if self.llm_name == "openai":
                return await self.call_openai(intent)
            elif self.llm_name == "anthropic":
                return await self.call_anthropic(intent)
            elif self.llm_name == "gemini":
                return await self.call_gemini(intent)
            elif self.llm_name == "qwen":
                return await self.call_qwen(intent)
                
        except Exception as e:
            self.logger.error(f"Error processing intent: {e}")
            raise

    async def cleanup(self):
        """Clean up resources (close streams & session)."""
        try:
            self.logger.info("Shuting down MCP connection.")
            await self.exit_stack.aclose()
        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")

    async def call_openai(self, intent: str):
        user_intent = {"role": "user", "content": intent}
        self.messages.append(user_intent)
        results_log = {}
        try:
            response = self.llm.responses.create(
                model=self.llm_model,
                input=self.messages,
                tools=self.tools
            )
            results_log = {"intent": intent}
            self.logger.info(f"Assistant response: {response}")
            for item in response.output:
                if item.type == "message":
                    assistant_message = {
                        "role": "assistant",
                        "content": item.content[0].text
                    }
                    self.messages.append(assistant_message)
                    self.logger.info("No tool_call found; returning text response.")
                    results_log = {"intent": intent, "intent_processing": str(response)}
                    return assistant_message
                elif item.type == "function_call":
                    tool_name = item.name
                    tool_args = item.arguments
                    call_id = item.call_id
        
                    self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")
                    try:
                        tool_result = await self.session.call_tool(
                            tool_name, json.loads(tool_args)
                        )
                        self.logger.info(f"Tool result: {tool_result}")
                        results_log = {"intent": intent, "intent_processing": str(response), "tool_call": str(tool_result)}
                    except Exception as e:
                        error_msg = f"Tool execution failed for {tool_name}: {str(e)}"
                        self.logger.error(error_msg)
                        raise Exception(error_msg)

                    if getattr(tool_result, 'content', None):
                            block = tool_result.content[0]
                            tool_output = getattr(block, 'text', None) or str(block)
                            self.messages.append({
                                "role": "user",
                                "content": f"Tool {tool_name} output (call_id={call_id}):\n{tool_output}"
                            })
                            slice = json.loads(tool_output)['message']
                            
                            self.logger.info("Setting policy type.")
                            slice_type_response = self.llm.responses.create(
                                model=self.llm_model,
                                input=intent,
                                instructions=self.instructions
                            )
                            slice_type = None
                            for item in slice_type_response.output:
                                if item.type == "message":
                                    slice_type = item.content[0].text
                            payload = json.dumps({
                                "sliceDescription": slice,
                                "sliceType": slice_type
                            })
                            policy = await self.slice_request(payload)
                            results_log = {"intent": intent, "intent_processing": str(response), "tool_call": str(tool_result), "type_definition": str(slice_type_response), "policy": policy.text}
                            return policy.text
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise
        finally:
            try:
                self.logger.info("Saving results file.")
                save_results(results_log,self.result_file_path)
                self.logger.info("File saved successfully.")
            except Exception as e:
                self.logger.error(f"Error saving results file: {e}")
                raise

    async def call_anthropic(self, intent: str):
        user_intent = {"role": "user", "content": intent}
        self.messages.append(user_intent)
        self._repair_anthropic_messages()
        results_log = {}
        try:
            response = self.llm.messages.create(
                    model=self.llm_model,
                    max_tokens=1000,
                    messages=self.messages,
                    tools=self.tools,
                )
            results_log = {"intent": intent}
            self.logger.info(f"Assistant response: {response}")
            if response.content[0].type == "text" and len(response.content) == 1:
                assistant_message = {
                    "role": "assistant",
                    "content": response.content[0].text,
                }
                self.messages.append(assistant_message)
                results_log = {"intent": intent, "tool_call": str(response)}
                self.logger.info("No tool_call found; returning text response.")  
                return assistant_message
            
            assistant_message = {
                "role": "assistant",
                "content": response.to_dict()["content"],
            }
            self.messages.append(assistant_message)  

            for content in response.content:
                if content.type == "tool_use":
                    tool_name = content.name
                    tool_args = content.input
                    tool_use_id = content.id
                    
                    self.logger.info(
                        f"Calling tool {tool_name} with args {tool_args}"
                    )
                    
                    try:
                        tool_result = await self.session.call_tool(tool_name, tool_args)
                        results_log = {"intent": intent, "intent_processing": str(response), "tool_call": str(tool_result)}
                        self.logger.info(f"Tool {tool_name} result: {tool_result}...")
                    except Exception as e:
                        self.logger.error(f"Error calling tool {tool_name}: {e}")
                        raise
                    
                    tool_output=tool_result.content[0].text
                    self.messages.append({
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": tool_output,
                            },
                        ],
                    })
                    slice = json.loads(tool_output)['message']
                    
                    self.logger.info("Setting policy type.")

                    intent_type = f"intent:{intent}{self.instructions}"
                    slice_type_response = self.llm.messages.create(
                        model=self.llm_model,
                        max_tokens=1000,
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": intent_type}
                                ],
                            }
                        ],
                    )
                    slice_type = slice_type_response.content[0].text
                    payload = json.dumps({
                                "sliceDescription": slice,
                                "sliceType": slice_type
                            })
                    policy = await self.slice_request(payload)
                    results_log = {"intent": intent, "intent_processing": str(response), "tool_call": str(tool_result), "type_definition": str(slice_type_response), "policy": policy.text}
                    return policy.text
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise
        finally:
            try:
                self.logger.info("Saving results file.")
                save_results(results_log,self.result_file_path)
                self.logger.info("File saved successfully.")
            except Exception as e:
                self.logger.error(f"Error saving results file: {e}")
                raise

    async def call_qwen(self, intent: str):
        user_intent = {"role": "user", "content": intent}
        self.messages.append(user_intent)
        results_log = {"intent": intent}
        try:
            response = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=self.messages,
                tools=self.tools,
            )

            self.logger.info(f"Assistant response: {response}")
            
            message = response.choices[0].message
            
            results_log = {"intent": intent, "intent_processing": str(response)}
            
            if not message.tool_calls:
                assistant_message = {"role": "assistant", "content": message.content or ""}
                self.messages.append(assistant_message)
                self.logger.info("No tool_call found; returning text response.")
                return assistant_message

            for item in message.tool_calls:
                tool_name = item.function.name
                tool_args = item.function.arguments
                call_id = item.id
                self.logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                try:
                    tool_result = await self.session.call_tool(
                        tool_name, json.loads(tool_args)
                    )
                    self.logger.info(f"Tool result: {tool_result}")
                    results_log = {"intent": intent, "intent_processing": str(response), "tool_call": str(tool_result)}
                except Exception as e:
                    error_msg = f"Tool execution failed for {tool_name}: {str(e)}"
                    self.logger.error(error_msg)
                    raise Exception(error_msg)

   
                if getattr(tool_result, "content", None):
                    block = tool_result.content[0]
                    tool_output = getattr(block, "text", None) or str(block)
                    self.messages.append({
                        "role": "user",
                        "content": f"Tool {tool_name} output (call_id={call_id}):\n{tool_output}"
                    })
                    slice = json.loads(tool_output)["message"]

                    self.logger.info("Setting policy type.")
                    slice_type_response = self.llm.chat.completions.create(
                        model=self.llm_model,
                        messages=[
                            {"role": "system", "content": self.instructions},
                            {"role": "user", "content": intent},
                        ],
                    )
                    slice_type = slice_type_response.choices[0].message.content

                    payload = json.dumps({
                        "sliceDescription": slice,
                        "sliceType": slice_type,
                    })
                    policy = await self.slice_request(payload)

                    results_log = {
                        "intent": intent,
                        "intent_processing": str(response),
                        "tool_call": str(tool_result),
                        "type_definition": str(slice_type_response),
                        "policy": policy.text,
                    }
                    return policy.text

        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise
        finally:
            try:
                self.logger.info("Saving results file.")
                save_results(results_log, self.result_file_path)
                self.logger.info("File saved successfully.")
                self.messages.clear()
            except Exception as e:
                self.logger.error(f"Error saving results file: {e}")
                raise

    def _message_has_block(self, message: dict, block_type: str) -> bool:
        content = message.get("content")
        if not isinstance(content, list):
            return False
        for block in content:
            if isinstance(block, dict) and block.get("type") == block_type:
                return True
        return False

    def _repair_anthropic_messages(self):
        if self.llm_name != "anthropic" or not self.messages:
            return
        repaired = []
        pending_tool_use = False
        for message in self.messages:
            if pending_tool_use:
                if self._message_has_block(message, "tool_result"):
                    pending_tool_use = False
                else:
                    if repaired:
                        repaired.pop()
                        self.logger.warning(
                            "Removed assistant tool_use without tool_result to satisfy Claude API ordering."
                        )
                    pending_tool_use = False
            repaired.append(message)
            if message.get("role") == "assistant" and self._message_has_block(message, "tool_use"):
                pending_tool_use = True
        if pending_tool_use and repaired:
            repaired.pop()
            self.logger.warning(
                "Dropped dangling assistant tool_use with no tool_result to prevent Claude API errors."
            )
        if repaired != self.messages:
            self.messages = repaired
            
    async def call_gemini(self, intent: str):
        user_intent = types.Content(role="user", parts=[types.Part(text=intent)])
        self.messages.append(user_intent)
        results_log = {}
        try:
            # Use dedicated typed contents for GenAI
            response = self.llm.models.generate_content(
                model=self.llm_model,
                contents=self.messages,
                config=self.tools,
            )
            results_log = {"intent": intent}
            self.logger.info(f"Assistant response: {response}")
            for cand in getattr(response, "candidates", []):
                for part in getattr(getattr(cand, "content", None), "parts", []) or []:
                    if getattr(part, "function_call", None):
                        function_call = part.function_call
                        break
                    else: 
                        assistant_message = {
                            "role": "assistant",
                            "content": getattr(cand, "content", None)
                        }
                        self.logger.info("No tool_call found; returning text response.")
                        results_log = {"intent": intent, "tool_call": str(response)}
                        return assistant_message
                    
            if function_call:
                tool_name = function_call.name
                tool_args = function_call.args
                self.logger.info(
                    f"Calling tool {tool_name} with args {tool_args}"
                )
                try:
                    tool_result = await self.session.call_tool(tool_name, tool_args)
                    results_log = {"intent": intent, "intent_processing": str(response), "tool_call": str(tool_result)}
                    self.logger.info(f"Tool result: {tool_result}")
                except Exception as e:
                    self.logger.error(f"Error calling tool {tool_name}: {e}")
                    raise
                
                tool_output = tool_result.content[0].text
            
                slice = json.loads(tool_output)['message']

                self.logger.info("Setting policy type.")
                slice_type_response = self.llm.models.generate_content(
                    model=self.llm_model,
                    contents=user_intent,
                    config=types.GenerateContentConfig( system_instruction=self.instructions)
                )

                for cand in getattr(slice_type_response, "candidates", []):
                    for part in getattr(getattr(cand, "content", None), "parts", []) or []:
                        if getattr(part, "text", None):
                           slice_type = part.text

                payload = json.dumps({
                                "sliceDescription": slice,
                                "sliceType": slice_type
                            })
                
                policy = await self.slice_request(payload)
                results_log = {"intent": intent, "intent_processing": str(response), "tool_call": str(tool_result), "type_definition": str(slice_type_response), "policy": policy.text}
                return policy.text
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}")
            raise
        finally:
            try:
                self.logger.info("Saving results file.")
                save_results(results_log,self.result_file_path)
                self.logger.info("File saved successfully.")
            except Exception as e:
                self.logger.error(f"Error saving results file: {e}")
                raise
        
    async def slice_request(self, payload : str):
        try:
            self.logger.info(f"Creating policy instance: {payload}")
            policy = requests.post(
                f"{self.rapp}/create_policy",
                json=json.loads(payload),
            )
            self.logger.info(f"Policy instance:{policy.text}")
            
            return policy
        except Exception as e:
            self.logger.error(f"Error creating policy instance.: {str(e)}")
            raise  

    @staticmethod
    def get_schema(tool):
        empty = {"type": "object", "properties": {}}
        for attr in ("inputSchema", "input_schema", "parameters"):
            value = getattr(tool, attr, None)
            if value is None and isinstance(tool, dict):
                value = tool.get(attr)
            if value:
                schema = dict(value)
                schema.setdefault("type", "object")
                schema.setdefault("properties", {})
                return schema
        return empty
