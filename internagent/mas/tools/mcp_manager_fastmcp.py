"""
MCP Server Connection Manager (FastMCP 2.0 Prototype)

This is a prototype implementation using FastMCP 2.0 client library.
Provides the same interface as the original MCPManager but with simplified
connection handling and better transport auto-detection.
"""

import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport, SSETransport

logger = logging.getLogger(__name__)

# Suppress SSE ping warnings from the underlying MCP SDK
# FastMCP's SSETransport uses mcp.client.sse internally, which logs these
sse_logger = logging.getLogger('mcp.client.sse')
sse_logger.addFilter(lambda record: 'Unknown SSE event: ping' not in record.getMessage())


class MCPManagerFastMCP:
    """
    Manager for MCP (Model Context Protocol) server connections using FastMCP 2.0.

    Key improvements over original implementation:
    - Automatic transport detection (stdio/SSE/HTTP)
    - Simpler connection management (no manual AsyncExitStack)
    - Built-in handling of SSE ping events (no warnings!)
    - Better error handling and connection resilience
    - OAuth support for authenticated servers

    Handles:
    - Auto-discovery of local MCP servers
    - Connection management for local (stdio) and remote (SSE/HTTP) servers
    - Tool listing and execution across all connected servers
    - Tool name to server session mapping
    """

    def __init__(self):
        """Initialize the MCP manager."""
        self.clients: Dict[str, Any] = {}  # server_id -> FastMCP Client
        self.tool_to_server: Dict[str, str] = {}  # tool_name -> server_id
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
                - url: Server URL (FastMCP auto-detects SSE/HTTP)
                - headers: Optional HTTP headers (for authentication)
                - auth: Optional authentication method ('oauth', 'bearer', etc.)
        """
        if self._initialized:
            logger.warning("MCPManagerFastMCP already initialized")
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
        logger.info(f"MCPManagerFastMCP initialized with {len(self.clients)} server(s)")
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

        FastMCP auto-detects transport based on file extension:
        - .py files → Python stdio transport
        - .js files → Node.js stdio transport

        Args:
            server_id: Unique identifier for this server
            script_path: Path to the server script (.py or .js)
        """
        logger.info(f"Connecting to local MCP server: {server_id}")

        try:
            # FastMCP Client auto-detects transport!
            # No need for manual StdioServerParameters or transport setup
            client = Client(script_path)

            # Initialize connection (FastMCP handles context management)
            await client.__aenter__()

            # Store client
            self.clients[server_id] = client

            # List and register tools from this server
            await self._register_server_tools(server_id, client)

            logger.info(f"Successfully connected to local server: {server_id}")

        except Exception as e:
            logger.error(f"Failed to connect to server {server_id}: {e}")
            raise

    async def _connect_remote_server(self, config: Dict[str, Any]):
        """
        Connect to a remote MCP server using FastMCP transports.

        Supports:
        - StreamableHttpTransport (recommended) - for /mcp endpoints
        - SSETransport (legacy) - for /sse endpoints
        - Auto-detection when no custom headers needed
        - Custom headers for authentication

        Args:
            config: Server configuration containing:
                - id: Unique server identifier
                - url: Server URL endpoint
                - headers: Optional dict of HTTP headers (default: {})
                - auth: Optional authentication (e.g., BearerAuth instance)
        """
        server_id = config['id']
        url = config['url']
        headers = config.get('headers', {})
        auth = config.get('auth', None)

        logger.info(f"Connecting to remote MCP server: {server_id} at {url}")

        try:
            # Per FastMCP docs: Client auto-detects transport from URL
            # BUT: Custom headers require explicit transport creation
            if headers:
                # Must create transport explicitly to pass custom headers
                from urllib.parse import urlparse
                url_path = urlparse(url).path.lower()

                # Detect transport type from URL path pattern
                if url_path.endswith('/sse') or '/sse/' in url_path:
                    logger.debug(f"Creating SSETransport with headers for {url}")
                    transport = SSETransport(url=url, headers=headers)
                elif url_path.endswith('/mcp') or '/mcp/' in url_path:
                    logger.debug(f"Creating StreamableHttpTransport with headers for {url}")
                    transport = StreamableHttpTransport(url=url, headers=headers)
                else:
                    # Unknown endpoint - try auto-detect (headers may not work)
                    logger.warning(f"Unknown endpoint pattern in {url}, headers may not be sent")
                    transport = url

                client = Client(transport, auth=auth) if auth else Client(transport)
            else:
                # No custom headers - use FastMCP's auto-detection
                logger.debug(f"Using auto-detection for {url}")
                client = Client(url, auth=auth) if auth else Client(url)

            await client.__aenter__()
            self.clients[server_id] = client
            await self._register_server_tools(server_id, client)
            logger.info(f"Successfully connected to {server_id}")

        except Exception as e:
            logger.error(f"Failed to connect to remote server {server_id}: {e}")
            logger.debug(f"Server config: {config}", exc_info=True)
            raise

    async def _register_server_tools(self, server_id: str, client: Client):
        """
        Register all tools from a connected server.

        Args:
            server_id: Server identifier
            client: Connected FastMCP client
        """
        try:
            # FastMCP provides simple list_tools() method
            tools = await client.list_tools()

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

        for server_id, client in self.clients.items():
            try:
                tools = await client.list_tools()

                for tool in tools:
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

        client = self.clients.get(server_id)
        if not client:
            raise RuntimeError(f"Server client not found: {server_id}")

        # Remove 'mcp_' prefix for actual MCP call
        actual_tool_name = tool_name[4:] if tool_name.startswith("mcp_") else tool_name

        try:
            logger.info(f"Executing MCP tool: {tool_name} on server {server_id}")

            # FastMCP provides simple call_tool() method
            result = await client.call_tool(actual_tool_name, arguments)

            # Extract content from MCP result
            # FastMCP result format is consistent with MCP SDK
            if hasattr(result, 'content'):
                if isinstance(result.content, list) and len(result.content) > 0:
                    # Return first content item's text
                    first_content = result.content[0]
                    if hasattr(first_content, 'text'):
                        return first_content.text
                    return str(first_content)
                return str(result.content)

            return str(result)

        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            raise RuntimeError(f"Tool execution failed: {e}")

    async def cleanup(self):
        """Clean up all server connections."""
        logger.info("Cleaning up MCP connections...")

        # FastMCP Client handles cleanup via __aexit__
        for server_id, client in list(self.clients.items()):
            try:
                await client.__aexit__(None, None, None)
                logger.debug(f"Closed connection to {server_id}")
            except Exception as e:
                logger.warning(f"Error closing connection to {server_id}: {e}")

        self.clients.clear()
        self.tool_to_server.clear()
        self._initialized = False
        logger.info("MCP cleanup complete")


# Convenience functions for comparison with original implementation
async def create_fastmcp_manager(
    local_servers_dir: Optional[str] = None,
    remote_servers: Optional[List[Dict[str, Any]]] = None
) -> MCPManagerFastMCP:
    """
    Create and initialize a FastMCP-based manager.

    This is the simplified API compared to the original implementation.
    """
    manager = MCPManagerFastMCP()
    await manager.initialize(local_servers_dir, remote_servers)
    return manager
