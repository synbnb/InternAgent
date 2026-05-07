# tools/query2smiles.py

import logging
import requests
from typing import Dict, Any
from urllib.parse import quote

logger = logging.getLogger(__name__)


def is_smiles(text: str) -> bool:
    """Check if text is a valid SMILES string"""
    import re
    smiles_pattern = r'^[A-Za-z0-9@+\-\[\]\(\)=#$:/\\\.%]+$'
    return bool(re.match(smiles_pattern, text))


def is_multiple_smiles(text: str) -> bool:
    """Check if text contains multiple SMILES strings"""
    # 检查是否有多个分子（通常用.分隔）
    return '.' in text and not text.startswith('.') and not text.endswith('.')


def pubchem_query2smiles(query: str) -> str:
    """Query PubChem for SMILES by molecule name"""
    # URL encode the query to handle special characters
    encoded_query = quote(query)
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{encoded_query}/property/CanonicalSMILES/JSON"
    
    logger.debug(f"Querying PubChem URL: {url}")
    
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    
    data = response.json()
    logger.debug(f"PubChem response: {data}")
    
    # Handle different possible response structures
    if 'PropertyTable' in data and 'Properties' in data['PropertyTable']:
        properties = data['PropertyTable']['Properties']
        if properties and len(properties) > 0:
            prop = properties[0]
            
            # Try different SMILES fields in order of preference
            for smiles_field in ['CanonicalSMILES', 'IsomericSMILES', 'ConnectivitySMILES']:
                if smiles_field in prop:
                    logger.info(f"Found {smiles_field}: {prop[smiles_field]}")
                    return prop[smiles_field]
    
    # If structure is different, try to find SMILES in the response
    raise ValueError(f"Could not find any SMILES field in PubChem response: {data}")


async def query2smiles(query: str) -> Dict[str, Any]:
    """
    Query molecule name and return SMILES string from PubChem.
    
    Args:
        query: Molecule name to query
        
    Returns:
        Dictionary containing:
            - success: Boolean indicating if query was successful
            - smiles: SMILES string if found
            - error: Error message if query failed
            - message: Human-readable success message
    """
    logger.info(f"Query2SMILES: Querying molecule name: {query}")
    
    # Check if input is already SMILES
    if is_smiles(query):
        if is_multiple_smiles(query):
            return {
                "success": False,
                "error": "Multiple SMILES strings detected, input one molecule at a time."
            }
    
    # Query PubChem
    try:
        smi = pubchem_query2smiles(query)
        logger.info(f"Found SMILES from PubChem: {smi}")
        
        return {
            "success": True,
            "smiles": smi,
            "message": f"Successfully converted '{query}' to SMILES: {smi}"
        }
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            error_msg = f"Molecule '{query}' not found in PubChem database"
        else:
            error_msg = f"PubChem API error (HTTP {e.response.status_code}): {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }
        
    except requests.exceptions.RequestException as e:
        error_msg = f"Network error when querying PubChem: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }
        
    except Exception as e:
        error_msg = f"Failed to find SMILES for '{query}': {str(e)}"
        logger.error(f"PubChem query failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": error_msg
        }


# Tool definition for agent
QUERY2SMILES_TOOL = {
    "type": "function",
    "function": {
        "name": "query2smiles",
        "description": (
            "Convert a molecule name to its SMILES (Simplified Molecular Input Line Entry System) representation. "
            "This tool queries PubChem database to find the SMILES string for a given molecule name. "
            "Only query with one specific molecule name at a time. "
            "Examples: 'aspirin', 'caffeine', 'ethanol', 'benzene', 'glucose'"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The name of the molecule to search for. "
                        "Should be a specific chemical name (e.g., 'aspirin', 'caffeine', 'ethanol'). "
                        "Do not input SMILES strings or multiple molecule names."
                    )
                }
            },
            "required": ["query"]
        }
    }
}
