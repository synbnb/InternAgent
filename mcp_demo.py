import os
import asyncio
import json
import logging
import sys
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

from internagent.mas.agents.base_agent import BaseAgent
from internagent.mas.models.base_model import BaseModel
from internagent.mas.models.openai_model import OpenAIModel
from internagent.mas.tools import get_registry, init_mcp_tools, cleanup_mcp, init_tools
from internagent.mas.tools.utils import get_related_tools

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



# Configuration (normally loaded from YAML)
# Test CONFIG 1: Both local and remote tools enabled
CONFIG = {
    "tools": {
        "local": True,  # Enable local function-based tools
        "remote": [
            {
                "id": "wiki",
                "url": "https://mcp.deepwiki.com/sse",
                "headers": {}
            },
            {
                "id": "tool_universe",
                "url": "http://localhost:52345/sse",
                "headers": {}
            }
        ]
    },
    "agents": [
        {
            "name": "full_access_agent",
            "model": "gpt-4o",
            "temperature": 0.7
            # No allowed_tools means all tools available
        },
    ]
}

# Test CONFIG 2: Only remote MCP tools, local tools disabled
CONFIG_NO_LOCAL = {
    "tools": {
        "local": False,  # Disable local function-based tools
        "remote": [
            {
                "id": "wiki",
                "url": "https://mcp.deepwiki.com/sse",
                "headers": {}
            }
        ]
    },
    "agents": [
        {
            "name": "mcp_only_agent",
            "model": "gpt-4o",
            "temperature": 0.7
        },
    ]
}

# Test CONFIG 3: Only local tools, no remote MCP servers
CONFIG_LOCAL_ONLY = {
    "tools": {
        "local": True,
        "remote": []
    },
    "agents": [
        {
            "name": "local_only_agent",
            "model": "gpt-4o",
            "temperature": 0.7
        },
    ]
}


class MyAgent(BaseAgent):
    """Assistant agent with support for both function-based and MCP tools"""
    
    def __init__(self, model: BaseModel, config: Dict[str, Any]):
        super().__init__(model, config)
        
        # Get tool registry
        self.tool_registry = get_registry()
        
        # Get allowed tools list from config (whitelist mode)
        self.allowed_tools = config.get("allowed_tools", None)
        
        logger.info(f"MyAgent '{self.name}' initialized")
        if self.allowed_tools:
            logger.info(f"Allowed tools: {self.allowed_tools}")
        else:
            logger.info(f"All tools available")
    
    async def _execute_tool(self, function_name: str, function_args: Dict[str, Any]) -> Any:
        """Execute tool function (routes to either function-based or MCP tool)"""
        logger.info(f"Executing tool: {function_name}")
        logger.debug(f"Arguments: {json.dumps(function_args, ensure_ascii=False)}")
        
        try:
            result = await self.tool_registry.execute(function_name, **function_args)
            logger.info(f"Tool execution successful: {function_name}")
            logger.info(f"Tool Call Result: {result}")
            return result
        except Exception as e:
            logger.error(f"Tool execution failed for {function_name}: {e}")
            raise
    
    async def get_registered_tools(self) -> list:
        """Get all tool definitions in OpenAI format (with permission filtering applied)"""
        return await self.tool_registry.get_all_definitions(
            allowed_tools=self.allowed_tools
        )
    
    async def execute(self, context: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent task"""
        query = context.get("query", "")
        
        if not query:
            return {"status": "error", "error": "No query provided"}
        
        logger.info(f"Processing query: {query}")
        
        try:
            tools = await self.get_registered_tools()
            logger.info(f"Available tools: {len(tools)}")


            # Retrieve related tools
            related_tools = get_related_tools(query, tools)
            logger.info(f"Related tools found: {len(related_tools)}")
                
            response = await self._call_model_with_tools(
                system_prompt="",
                prompt=query,
                tools=related_tools,
                temperature=params.get("temperature", 0.7),
                max_iterations=params.get("max_iterations", 10),
                max_tool_calls=params.get("max_tool_calls", 20)
            )
            
            return {
                "status": "success",
                "answer": response["content"],
                "tool_calls": response["tool_calls_made"],
                "iterations": response["iterations"]
            }
        
        except Exception as e:
            logger.error(f"Error during execution: {e}")
            return {"status": "error", "error": str(e)}


async def main():
    print("\n" + "="*60)
    print("MyAgent Demo - Function-based + MCP Tools")
    print("="*60)

    # Switch between different configs to test:
    # - CONFIG: both local and remote tools
    # - CONFIG_NO_LOCAL: only remote MCP tools
    # - CONFIG_LOCAL_ONLY: only local function tools
    active_config = CONFIG_LOCAL_ONLY  # Change this to test different configurations

    try:
        # Step 1: Initialize tools from config
        print("\n" + "="*60)
        print("TOOL INITIALIZATION")
        print("="*60)

        tools_config = active_config.get("tools", {})

        # Initialize local function-based tools if enabled
        if tools_config.get("local", True):
            print("\n[✓] Loading local function-based tools...")
            init_tools()
            print("    Local tools initialized successfully")
        else:
            print("\n[✗] Local tools DISABLED by configuration")

        # Initialize remote MCP tools if configured
        remote_servers = tools_config.get("remote", [])
        if remote_servers:
            print(f"\n[✓] Found {len(remote_servers)} remote MCP server(s) in config:")
            for server in remote_servers:
                print(f"    - {server['id']}: {server['url']}")
            await init_mcp_tools(remote_servers=remote_servers)
        else:
            print("\n[✗] No remote MCP servers configured")

        # Get registry statistics
        registry = get_registry()
        print("\n" + "="*60)
        print("TOOL REGISTRY STATISTICS")
        print("="*60)
        print(f"Total tools loaded: {registry.total_tool_count()}")
        print(f"  - Function tools: {len(registry)}")
        print(f"  - MCP tools: {registry.total_tool_count() - len(registry)}")
        print(f"\nRegistered tool names:")
        for name in sorted(registry.get_all_names()):
            print(f"  - {name}")
        
        # Step 2: Initialize model
        print("\n" + "="*60)
        print("MODEL & AGENT INITIALIZATION")
        print("="*60)
        print("\nInitializing model...")
        model = OpenAIModel(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-4o",
            temperature=0.7
        )

        # Step 3: Create agent from config
        agent_config = active_config["agents"][0]  # Use first agent config
        
        print(f"\nCreating agent: {agent_config['name']}")
        agent = MyAgent(
            model=model,
            config={
                "name": agent_config["name"],
                "allowed_tools": agent_config.get("allowed_tools", None)
            }
        )
        
        tools = await agent.get_registered_tools()
        print(f"Agent has {len(tools)} tools available")
        if agent_config.get("allowed_tools"):
            print(f"Allowed tools: {agent_config['allowed_tools']}")
        
        # Step 4: Run test queries
        test_queries = [
            "Design a novel, metal-free, and inherently heterogeneous organocatalyst (e.g., based on Covalent Organic Frameworks, Porous Organic Polymers, or supported active sites) engineered for integrated Carbon Capture and Utilization (CCU) processes, focusing on the efficient chemical conversion of carbon dioxide ($CO_2$) into high-value feedstocks (such as cyclic carbonates or formate/methanol). The critical design challenge is to ensure the catalyst exhibits high activity, selectivity, and stability under realistic flue gas conditions, specifically at low $CO_2$ partial pressures (e.g., 0.1-1 bar), mild temperatures (ambient to 80°C), and in the presence of impurities like water, oxygen, and trace $SO_x$/$NO_x$. The catalyst must be easily separable and highly recyclable (>10 cycles) and should incorporate an innovative bifunctional active site (e.g., a synergistic Lewis base/H-bond donor pair or embedded frustrated Lewis pairs) to co-activate both $CO_2$ and the co-substrate, ideally achieving $CO_2$ adsorption and conversion at a single site. Please use the available computational tools to propose a detailed catalyst design, including the selection of building blocks, synthesis routes, and mechanistic pathways for $CO_2$ activation and conversion under the specified conditions. Research existing organocatalysts using `query2smiles` to obtain their SMILES structures. Generate structural variants using `modify_mol`.Verify novelty with `patent_check`. propose a novel organocatalyst with detailed rationale for its design and expected performance. Ensure all steps are documented and justified.Use the 'calculate' when numerical computations are needed. And you may search literature with to support your design choices.",
            # "Propose a comprehensive computational strategy for the development of a generative energy-based framework (GEARS) capable of learning joint representations from single-cell multi-omics data (scRNA-seq + scATAC-seq) to predict transcriptional outcomes after genetic or chemical perturbations. This strategy must detail the practical implementation steps, beginning with the identification and acquisition of requisite large-scale datasets using resources like get_cellxgene_census_info. The proposal must then define the complete data preprocessing pipeline, specifying how genomic interval operations (using get_pybedtools_info or get_pyranges_info) will be used to map scATAC-seq accessibility peaks to gene features derived from genomic annotations (via get_pyensembl_info). Furthermore, the strategy must incorporate the inference of biological priors, such as constructing a gene regulatory network (using get_arboreto_info), to constrain the model. Finally, it must outline a rigorous validation protocol that leverages differential expression analysis (via get_pydeseq2_info) and pathway enrichment (via get_gseapy_info) to systematically compare the model's predicted cellular states against established perturbation ground truths.",
            # "Design a novel organocatalyst for enhancing CO2 conversion in carbon capture processes. ",
            # "What the latest version of numpy?",
            #"What's the weather forecast for San Francisco (latitude 37.7749, longitude -122.4194)?",
            #"Calculate 25 * 4 + 10",
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"Test {i}/{len(test_queries)}")
            print(f"Query: {query}")
            print(f"{'-'*60}")
            
            result = await agent.execute(
                context={"query": query},
                params={"temperature": 0.7}
            )
            
            if result["status"] == "success":
                print(f"\nAnswer: {result['answer']}")
                print(f"Tool calls: {len(result['tool_calls'])}")
                if result['tool_calls']:
                    print("Tools used:")
                    for call in result['tool_calls']:
                        print(f"  - {call['function']['name']}")
            else:
                print(f"\nError: {result['error']}")
        
        print("\n" + "="*60)
        print("Demo completed")
        print("="*60 + "\n")
    
    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        print(f"\nDemo failed: {e}\n")
    
    finally:
        print("\nCleaning up...")
        await cleanup_mcp()
        print("Cleanup complete\n")


if __name__ == "__main__":
    asyncio.run(main())
