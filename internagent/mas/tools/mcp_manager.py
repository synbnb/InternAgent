"""
MCP Server Connection Manager

Manages connections to multiple MCP servers (local and remote) and provides
a unified interface for tool discovery and execution.
"""

import os
import logging
import asyncio
from typing import Dict, List, Any, Optional
from pathlib import Path
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)


class MCPManager:
    """
    Manager for MCP (Model Context Protocol) server connections.
    
    Handles:
    - Auto-discovery of local MCP servers
    - Connection management for local (stdio) and remote (SSE/HTTP) servers
    - Tool listing and execution across all connected servers
    - Tool name to server session mapping
    """
    
    def __init__(self):
        """Initialize the MCP manager."""
        self.sessions: Dict[str, ClientSession] = {}  # server_id -> session
        self.tool_to_server: Dict[str, str] = {}  # tool_name -> server_id
        self.exit_stack = AsyncExitStack()
        self._initialized = False
        
    async def initialize(self, 
                        local_servers_dir: Optional[str] = None,
                        remote_servers: Optional[List[Dict[str, Any]]] = None):
        """
        Initialize all MCP server connections.
        
        Args:
            local_servers_dir: Directory containing local MCP server scripts.
                             Defaults to internagent/mas/tools/mcp/
            remote_servers: List of remote server configurations, each containing:
                - id: Unique identifier
                - url: Server URL (can be /sse or /mcp endpoint)
                - headers: Optional HTTP headers (for authentication)
                - protocol: Optional, 'sse' or 'http' (auto-detected if not specified)
        """
        if self._initialized:
            logger.warning("MCPManager already initialized")
            return
            
        # Auto-discover and connect local servers
        if local_servers_dir is None:
            current_dir = Path(__file__).parent
            local_servers_dir = current_dir / "mcp"
        else:
            local_servers_dir = Path(local_servers_dir)
            
        if local_servers_dir.exists():
            await self._discover_and_connect_local_servers(local_servers_dir)
        else:
            logger.warning(f"Local MCP servers directory not found: {local_servers_dir}")
        
        # Connect to remote servers if provided
        if remote_servers:
            logger.info(f"Attempting to connect to {len(remote_servers)} remote server(s)")
            for server_config in remote_servers:
                try:
                    await self._connect_remote_server(server_config)
                except Exception as e:
                    server_id = server_config.get('id', 'unknown')
                    logger.error(f"Failed to connect to remote server {server_id}: {e}")
                    logger.debug(f"Server config: {server_config}", exc_info=True)
        
        self._initialized = True
        logger.info(f"MCPManager initialized with {len(self.sessions)} server(s)")
        logger.info(f"Total MCP tools available: {len(self.tool_to_server)}")
        
    async def _discover_and_connect_local_servers(self, servers_dir: Path):
        """
        Discover and connect to all local MCP servers in the directory.
        
        Args:
            servers_dir: Directory to scan for *_server.py files
        """
        if not servers_dir.is_dir():
            logger.warning(f"Directory not found: {servers_dir}")
            return
            
        # Find all *_server.py files (exclude special files like __init__.py)
        all_py_files = list(servers_dir.glob("*.py"))
        server_files = [
            f for f in all_py_files
            if f.name.endswith('_server.py') and not f.name.startswith('_')
        ]
        
        if not server_files:
            logger.info(f"No MCP server files (*_server.py) found in {servers_dir}")
            if all_py_files:
                logger.info(f"Found {len(all_py_files)} .py files but none match *_server.py pattern")
            return
            
        logger.info(f"Found {len(server_files)} local MCP server(s): {[f.name for f in server_files]}")
        
        # Connection statistics
        connected = 0
        failed = 0
        
        for server_file in server_files:
            server_id = server_file.stem
            
            try:
                await self._connect_local_server(
                    server_id=server_id,
                    script_path=str(server_file.absolute())
                )
                connected += 1
            except Exception as e:
                failed += 1
                logger.error(f"Failed to connect to local server {server_id}: {e}")
        
        logger.info(f"Local server connection summary: {connected} connected, {failed} failed")
            
    async def _connect_local_server(self, server_id: str, script_path: str):
        """
        Connect to a local MCP server via stdio.
        
        Args:
            server_id: Unique identifier for this server
            script_path: Path to the server script (.py or .js)
        """
        logger.info(f"Connecting to local MCP server: {server_id}")
        
        # Determine command based on file extension
        if script_path.endswith('.py'):
            command = "python"
        elif script_path.endswith('.js'):
            command = "node"
        else:
            raise ValueError(f"Unsupported server script type: {script_path}")
        
        # Set up stdio transport
        server_params = StdioServerParameters(
            command=command,
            args=[script_path],
            env=None
        )
        
        try:
            # Create stdio transport
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            stdio, write = stdio_transport
            
            # Create and initialize session
            session = await self.exit_stack.enter_async_context(
                ClientSession(stdio, write)
            )
            await session.initialize()
            
            # Store session
            self.sessions[server_id] = session
            
            # List and register tools from this server
            await self._register_server_tools(server_id, session)
            
        except Exception as e:
            logger.error(f"Failed to connect to server {server_id}: {e}")
            raise
    
    def _detect_protocol_from_url(self, url: str) -> str:
        """
        Detect protocol type from URL.
        
        Args:
            url: Server URL
            
        Returns:
            'sse' or 'http'
        """
        url_lower = url.lower()
        
        # Check URL path
        if '/sse' in url_lower:
            return 'sse'
        elif '/mcp' in url_lower:
            return 'http'
        
        # Default to SSE (standard MCP protocol)
        return 'sse'
    
    async def _connect_remote_server(self, config: Dict[str, Any]):
        """
        Connect to a remote MCP server.
        
        Supports two protocols:
        1. SSE (Server-Sent Events) - standard MCP protocol
        2. Streamable HTTP - newer protocol for Cloudflare/OpenAI
        
        If protocol is not specified, auto-detects from URL or tries both.
        
        Args:
            config: Server configuration containing:
                - id: Unique server identifier
                - url: Server URL endpoint
                - headers: Optional dict of HTTP headers
                - protocol: Optional, 'sse' or 'http' (auto-detected if not specified)
        """
        server_id = config['id']
        url = config['url']
        headers = config.get('headers', {})
        protocol = config.get('protocol', None)
        
        logger.info(f"Connecting to remote MCP server: {server_id} at {url}")
        
        # Auto-detect protocol if not specified
        if protocol is None:
            protocol = self._detect_protocol_from_url(url)
            logger.info(f"Auto-detected protocol: {protocol}")
        
        # Try specified protocol first
        try:
            if protocol == 'http':
                await self._connect_via_http(server_id, url, headers)
            else:  # Default to SSE
                await self._connect_via_sse(server_id, url, headers)
            
            logger.info(f"Successfully connected to {server_id} via {protocol.upper()}")
            return
            
        except Exception as e:
            logger.warning(f"Failed to connect via {protocol.upper()}: {e}")
            
            # Try fallback protocol
            fallback_protocol = 'sse' if protocol == 'http' else 'http'
            logger.info(f"Attempting fallback to {fallback_protocol.upper()}...")
            
            try:
                # Adjust URL for fallback
                fallback_url = self._adjust_url_for_protocol(url, fallback_protocol)
                
                if fallback_protocol == 'http':
                    await self._connect_via_http(server_id, fallback_url, headers)
                else:
                    await self._connect_via_sse(server_id, fallback_url, headers)
                
                logger.info(f"Successfully connected to {server_id} via {fallback_protocol.upper()} (fallback)")
                return
                
            except Exception as fallback_error:
                logger.error(f"Fallback connection also failed: {fallback_error}")
                raise Exception(f"Failed to connect with both protocols: {protocol} and {fallback_protocol}")
    
    def _adjust_url_for_protocol(self, url: str, protocol: str) -> str:
        """
        Adjust URL endpoint for the specified protocol.
        
        Args:
            url: Original URL
            protocol: Target protocol ('sse' or 'http')
            
        Returns:
            Adjusted URL
        """
        # Replace endpoint if needed
        if protocol == 'sse':
            if '/mcp' in url:
                return url.replace('/mcp', '/sse')
        elif protocol == 'http':
            if '/sse' in url:
                return url.replace('/sse', '/mcp')
        
        return url
    
    async def _connect_via_sse(self, server_id: str, url: str, headers: Dict[str, str]):
        """
        Connect to remote server via SSE (Server-Sent Events).
        
        Args:
            server_id: Server identifier
            url: Server URL
            headers: HTTP headers
        """
        try:
            from mcp.client.sse import sse_client
        except ImportError:
            raise ImportError(
                "SSE client not available. Install with: pip install mcp"
            )
        
        logger.debug(f"Connecting via SSE to {url}")
        
        # Create SSE transport
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(url, headers=headers)
        )
        read_stream, write_stream = sse_transport
        
        # Create and initialize session
        session = await self.exit_stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        
        # Store session
        self.sessions[server_id] = session
        
        # Register tools
        await self._register_server_tools(server_id, session)

    async def _connect_via_http(self, server_id: str, url: str, headers: Dict[str, str]):
        """
        Connect to remote server via Streamable HTTP.
        
        This is a newer protocol that works with Cloudflare and OpenAI.
        Currently, the official MCP Python SDK does not have full support for this protocol.
        We attempt a fallback to SSE.
        
        Args:
            server_id: Server identifier
            url: Server URL
            headers: HTTP headers
        """
        logger.info(f"Attempting HTTP connection to {url}")
        
        # Check if http_client is available
        http_client_available = False
        try:
            from mcp.client import http as mcp_http
            if hasattr(mcp_http, 'http_client'):
                http_client_available = True
        except (ImportError, AttributeError):
            pass
        
        if not http_client_available:
            logger.warning(
                f"MCP HTTP client not available in current SDK version. "
                f"The Streamable HTTP protocol (/mcp endpoint) is not fully supported yet."
            )
            
            # Try to convert URL to SSE endpoint and use that instead
            if '/mcp' in url:
                sse_url = url.replace('/mcp', '/sse')
                logger.info(f"Attempting to use SSE endpoint instead: {sse_url}")
                await self._connect_via_sse(server_id, sse_url, headers)
            else:
                raise NotImplementedError(
                    f"Streamable HTTP protocol not supported for {server_id}. "
                    f"Please use the SSE endpoint (replace /mcp with /sse in the URL) or "
                    f"wait for MCP SDK to add HTTP support."
                )
            return
        
        # If http_client is available, use it
        try:
            from mcp.client.http import http_client
            
            logger.debug(f"Connecting via HTTP to {url}")
            
            # Create HTTP transport
            http_transport = await self.exit_stack.enter_async_context(
                http_client(url, headers=headers)
            )
            read_stream, write_stream = http_transport
            
            # Create and initialize session
            session = await self.exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            
            # Store session
            self.sessions[server_id] = session
            
            # Register tools
            await self._register_server_tools(server_id, session)
            
        except Exception as e:
            logger.error(f"HTTP connection failed: {e}")
            raise

    async def _register_server_tools(self, server_id: str, session: ClientSession):
        """
        Register all tools from a connected server.
        
        Args:
            server_id: Server identifier
            session: Connected client session
        """
        try:
            response = await session.list_tools()
            tools = response.tools
            
            for tool in tools:
                # Register tool with MCP prefix to avoid conflicts
                tool_name = f"mcp_{tool.name}"
                self.tool_to_server[tool_name] = server_id
                logger.info(f"  Registered tool: {tool_name}")
            
            logger.info(f"Connected to {server_id} with {len(tools)} tool(s)")
            
        except Exception as e:
            logger.error(f"Failed to register tools from {server_id}: {e}")
            raise
        
    async def list_all_tools(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions from all connected servers in OpenAI format.
        
        Returns:
            List of tool definitions in OpenAI function calling format
        """
        all_tools = []
        
        for server_id, session in self.sessions.items():
            try:
                response = await session.list_tools()
                
                for tool in response.tools:
                    # Convert MCP tool to OpenAI format
                    tool_name = f"mcp_{tool.name}"
                    
                    openai_tool = {
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "description": tool.description or "",
                            "parameters": tool.inputSchema
                        }
                    }
                    
                    all_tools.append(openai_tool)
                    
            except Exception as e:
                logger.error(f"Error listing tools from server {server_id}: {e}")
                
        return all_tools
    
    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute an MCP tool by name.
        
        Args:
            tool_name: Name of the tool (with 'mcp_' prefix)
            arguments: Tool arguments as dictionary
            
        Returns:
            Tool execution result
        """
        # Find which server owns this tool
        server_id = self.tool_to_server.get(tool_name)
        
        if not server_id:
            raise ValueError(f"MCP tool not found: {tool_name}")
        
        session = self.sessions.get(server_id)
        if not session:
            raise RuntimeError(f"Server session not found: {server_id}")
        
        # Remove 'mcp_' prefix for actual MCP call
        actual_tool_name = tool_name[4:] if tool_name.startswith("mcp_") else tool_name
        
        try:
            logger.info(f"Executing MCP tool: {tool_name} on server {server_id}")
            result = await session.call_tool(actual_tool_name, arguments)
            
            # Extract content from MCP result
            if hasattr(result, 'content'):
                if isinstance(result.content, list) and len(result.content) > 0:
                    # Return first content item's text
                    return result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                return str(result.content)
            
            return str(result)
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            raise RuntimeError(f"Tool execution failed: {e}")
    
    async def cleanup(self):
        """Clean up all server connections."""
        logger.info("Cleaning up MCP connections...")
        await self.exit_stack.aclose()
        self.sessions.clear()
        self.tool_to_server.clear()
        self._initialized = False
        logger.info("MCP cleanup complete")
