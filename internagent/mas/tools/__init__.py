"""
Tools Module 
"""

import logging
from typing import Optional, List, Dict, Any

from .tool_config import ToolRegistry, get_registry, register_tool
# from .mcp_manager import MCPManager
from .mcp_manager_fastmcp import MCPManagerFastMCP as MCPManager

# Import tool definitions and functions
from .search_tools.demo import PATENT_CHECK_TOOL, patent_check
from .search_tools.demo import CALCULATOR_TOOL_DEFINITION, calculate
from .search_tools.academic import ACADEMIC_SEARCH_TOOL, academic_search
from .sci_tools.chem.convertor import QUERY2SMILES_TOOL, query2smiles
from .sci_tools.chem.modify_mol import MODIFY_MOL_TOOL, modify_mol

logger = logging.getLogger(__name__)

# Global MCP manager instance
_mcp_manager = None
_mcp_initialized = False


def init_tools():
    """Initialize and register all function-based tools"""
    register_tool(CALCULATOR_TOOL_DEFINITION, calculate)
    register_tool(QUERY2SMILES_TOOL, query2smiles)
    register_tool(PATENT_CHECK_TOOL, patent_check)
    register_tool(MODIFY_MOL_TOOL, modify_mol)
    register_tool(ACADEMIC_SEARCH_TOOL, academic_search)
    logger.info("Function tools registered")


async def init_mcp_tools(
    local_servers_dir: Optional[str] = None,
    remote_servers: Optional[List[Dict[str, Any]]] = None
) -> MCPManager:
    """
    Initialize MCP tools (both local and remote).
    
    Should be called once during application startup, before creating any agents.
    
    Args:
        local_servers_dir: Directory for local MCP servers, defaults to tools/mcp/
        remote_servers: List of remote server configurations, each containing:
            - id: Unique server identifier
            - url: Server URL endpoint
            - headers: Optional dict of HTTP headers (for authentication)
    
    Returns:
        MCPManager instance
    """
    global _mcp_manager, _mcp_initialized
    
    if _mcp_initialized:
        logger.warning("MCP tools already initialized")
        return _mcp_manager
    
    if _mcp_manager is None:
        _mcp_manager = MCPManager()
    
    # Initialize MCP connections (both local and remote)
    await _mcp_manager.initialize(local_servers_dir, remote_servers)
    
    # Attach MCP manager to tool registry
    registry = get_registry()
    registry.set_mcp_manager(_mcp_manager)
    
    _mcp_initialized = True
    logger.info("MCP tools initialized and attached to registry")
    
    return _mcp_manager


def get_mcp_manager() -> Optional[MCPManager]:
    """Get global MCP manager instance"""
    global _mcp_manager
    if _mcp_manager is None:
        logger.warning("MCP manager not initialized. Call init_mcp_tools() first.")
    return _mcp_manager


def is_mcp_initialized() -> bool:
    """Check if MCP tools have been initialized"""
    return _mcp_initialized


async def cleanup_mcp():
    """Clean up MCP connections"""
    global _mcp_manager, _mcp_initialized
    if _mcp_manager:
        await _mcp_manager.cleanup()
        _mcp_manager = None
        _mcp_initialized = False
        logger.info("MCP connections cleaned up")


# Module initialization: Register function-based tools automatically on import
# Note: init_tools() should be called explicitly based on configuration
# init_tools()  # Commented out - now controlled by config


__all__ = [
    'ToolRegistry',
    'get_registry',
    'register_tool',
    'init_tools',
    'get_weather',
    'calculate',
    'academic_search',
    'init_mcp_tools',
    'get_mcp_manager',
    'is_mcp_initialized',
    'cleanup_mcp',
    'MCPManager'
]
