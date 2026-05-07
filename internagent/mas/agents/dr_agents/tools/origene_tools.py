from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
import asyncio

NEW_ROOT_URL = "http://106.14.54.82"    ### 评测benchmark请用此地址
# NEW_ROOT_URL = "http://106.15.24.29"

tool_packages = [
    "chembl_mcp",
    "kegg_mcp",
    "string_mcp",
    "search_mcp",
    "pubchem_mcp",
    "ncbi_mcp",
    "uniprot_mcp",
    "tcga_mcp",
    "ensembl_mcp",
    "ucsc_mcp",
    "fda_drug_mcp",
    "opentargets_mcp",
    "monarch_mcp",
    "clinicaltrials_mcp",
]

if NEW_ROOT_URL == "http://106.15.24.29":
    tool_packages.append("zhihuiya_mcp")
    tool_packages.append("dptech_mcp")

mcp_servers = {
    package: {
        "transport": "streamable_http",
        "url": f"{NEW_ROOT_URL}/mcp_index/{package}/mcp/",
    }
    for package in tool_packages
}


class OrigeneMCPToolClient:
    def __init__(self, mcp_servers: dict, specified_tools: list = None):
        self.mcp_servers = mcp_servers
        self.mcp_tools = None
        self.mcp_tool_map = {}
        self.available_tools = specified_tools

    async def initialize(self):
        """Initialize async components"""
        client = MultiServerMCPClient(self.mcp_servers)

        self.tool2source = {}
        for pkg_name in self.mcp_servers.keys():
            async with client.session(pkg_name) as session:
                tools = await load_mcp_tools(session)
                self.tool2source.update(
                    {tool.name: pkg_name.replace("_mcp", "") for tool in tools}
                )

        self.mcp_tools = await client.get_tools()
        if self.available_tools:
            self.mcp_tools = [
                tool for tool in self.mcp_tools if tool.name in self.available_tools
            ]
        self.mcp_tool_map = {tool.name: tool for tool in self.mcp_tools}
        print(f"MCP server connected! Found {len(self.mcp_tools)} tools")


available_tools = [
    'tavily_search',
    # 'jina_search',
    'gsea_search',
    'get_general_info_by_compound_name',
    'get_general_info_by_protein_or_gene_name',
    # 'get_disease_id_description_by_name',
]

async def connect_mcp():
    client = MultiServerMCPClient(mcp_servers)
    tools = await client.get_tools()

    # 过滤掉 jina 工具（临时禁用）
    filtered_tools = []
    for tool in tools:
        # if hasattr(tool, 'name') and tool.name == 'jina_search':
        #     print(f"⚠️  WARNING: jina_search tool is temporarily disabled. Skipping...")
        if hasattr(tool, 'name') and tool.name == 'get_disease_id_description_by_name':
            print(f"⚠️  WARNING: get_disease_id_description_by_name tool is temporarily disabled. Skipping...")
            continue
        if tool.name in available_tools:
            filtered_tools.append(tool)

    print(f"✅ MCP server connected! Found {len(filtered_tools)} tools (after filtering):")
    for tool in filtered_tools:
        if hasattr(tool, 'name'):
            print(f"  - {tool.name}")

    return filtered_tools


if __name__ == "__main__":
    asyncio.run(connect_mcp())
