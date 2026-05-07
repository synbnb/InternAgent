"""
Tool Configuration 
"""

from typing import Dict, Callable, Any, List, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)


class ToolRegistry:
    """Tool Registry - Unified management of functional tools and MCP tools"""
    
    def __init__(self):
        # Store tool functions: {tool_name: function}
        self._functions: Dict[str, Callable] = {}
        # Store tool definitions: {tool_name: definition}
        self._definitions: Dict[str, Dict[str, Any]] = {}
        # MCP manager (lazy injection)
        self._mcp_manager: Optional['MCPManager'] = None
    
    def register(self, definition: Dict[str, Any], function: Callable) -> None:
        """
        Register a functional tool
        
        Args:
            definition: Tool definition in OpenAI format
            function: Implementation function of the tool
        """
        tool_name = definition["function"]["name"]
        
        if tool_name in self._functions:
            logger.warning(f"Tool '{tool_name}' already registered, overwriting")
        
        self._functions[tool_name] = function
        self._definitions[tool_name] = definition
        
        logger.info(f"Registered function tool: {tool_name}")
    
    def set_mcp_manager(self, mcp_manager: 'MCPManager') -> None:
        """
        Set MCP manager
        
        Args:
            mcp_manager: MCP manager instance
        """
        self._mcp_manager = mcp_manager
        logger.info("MCP manager attached to ToolRegistry")
    
    def get_function(self, tool_name: str) -> Callable:
        """Get functional tool"""
        if tool_name not in self._functions:
            raise ValueError(f"Function tool '{tool_name}' not found")
        return self._functions[tool_name]
    
    async def get_all_definitions(self, allowed_tools: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Get all tool definitions in OpenAI format (functional + MCP)
        
        Args:
            allowed_tools: Whitelist of tool names, None means all tools are available
        
        Returns:
            List of tool definitions
        """
        all_definitions = []
        
        # 1. Add functional tools
        for tool_name, definition in self._definitions.items():
            if allowed_tools is None or tool_name in allowed_tools:
                all_definitions.append(definition)
        
        # 2. Add MCP tools
        if self._mcp_manager:
            try:
                mcp_tools = await self._mcp_manager.list_all_tools()
                for tool in mcp_tools:
                    tool_name = tool['function']['name']
                    if allowed_tools is None or tool_name in allowed_tools:
                        all_definitions.append(tool)
            except Exception as e:
                logger.error(f"Error getting MCP tools: {e}")
        
        return all_definitions
    
    def get_all_definitions_sync(self) -> List[Dict[str, Any]]:
        """
        Synchronously get all functional tool definitions (for backward compatibility)
        
        Returns:
            List of functional tool definitions
        """
        return list(self._definitions.values())
    
    def get_all_names(self, include_mcp: bool = True) -> List[str]:
        """
        Get all tool names
        
        Args:
            include_mcp: Whether to include MCP tool names
        
        Returns:
            List of tool names
        """
        names = list(self._functions.keys())
        
        if include_mcp and self._mcp_manager:
            mcp_names = list(self._mcp_manager.tool_to_server.keys())
            names.extend(mcp_names)
        
        return names
    
    async def execute(self, tool_name: str, **kwargs) -> Any:
        """
        Execute tool function (automatically route to functional or MCP)
        
        Args:
            tool_name: Tool name
            **kwargs: Tool parameters
        
        Returns:
            Tool execution result
        """
        # 1. Try to execute functional tools
        if tool_name in self._functions:
            function = self._functions[tool_name]
            if asyncio.iscoroutinefunction(function):
                return await function(**kwargs)
            else:
                return function(**kwargs)
        # 2. Try to execute MCP tool
        if self._mcp_manager and tool_name in self._mcp_manager.tool_to_server:
            return await self._mcp_manager.execute_tool(tool_name, kwargs)
        
        raise ValueError(f"Tool '{tool_name}' not found in function tools or MCP tools")
    
    def __len__(self) -> int:
        """Return the number of functional tools"""
        return len(self._functions)
    
    def total_tool_count(self) -> int:
        """Return total tool count (functional + MCP)"""
        count = len(self._functions)
        if self._mcp_manager:
            count += len(self._mcp_manager.tool_to_server)
        return count


# Global tool registry
_global_registry = ToolRegistry()


def get_registry() -> ToolRegistry:
    """Get global tool registry"""
    return _global_registry


def register_tool(definition: Dict[str, Any], function: Callable) -> None:
    """
    Register tool to global registry
    
    Args:
        definition: Tool definition
        function: Tool function
    """
    _global_registry.register(definition, function)
