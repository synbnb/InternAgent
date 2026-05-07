"""
Weather Tool - 天气查询工具函数
"""

import asyncio
import math
import logging
import molbloom
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

# 工具的元数据定义
CALCULATOR_TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "calculate",
        "description": "Perform mathematical calculations. Supports +, -, *, /, **, sqrt, sin, cos, etc.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to calculate, e.g., '2 + 3 * 4' or 'sqrt(16)'"
                }
            },
            "required": ["expression"]
        }
    }
}

# Tool definition for agent
PATENT_CHECK_TOOL = {
    "type": "function",
    "function": {
        "name": "patent_check",
        "description": (
            "Check if a molecule is patented by querying the SureChEMBL patent database. "
            "This tool determines whether a compound has been patented or is novel. "
            "You can input a single SMILES string or multiple SMILES strings separated by periods. "
            "Returns 'Patented' if the molecule is found in patents, or 'Novel' if not found."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": (
                        "SMILES string(s) to check for patent status. "
                        "Can be a single SMILES (e.g., 'CC(=O)OC1=CC=CC=C1C(=O)O') "
                        "or multiple SMILES separated by periods (e.g., 'CCO.CC(C)O.c1ccccc1'). "
                        "Each SMILES will be checked independently."
                    )
                }
            },
            "required": ["smiles"]
        }
    }
}

async def calculate(expression: str) -> Dict[str, Any]:
    """
    执行数学计算
    
    Args:
        expression: 数学表达式字符串
    
    Returns:
        包含计算结果的字典
    """
    logger.info(f"Calculating: {expression}")
    
    try:
        # 安全的数学计算环境
        safe_dict = {
            "sqrt": math.sqrt,
            "pow": math.pow,
            "abs": abs,
            "round": round,
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "log": math.log,
            "exp": math.exp,
            "pi": math.pi,
            "e": math.e,
        }
        
        # 执行计算
        result = eval(expression, {"__builtins__": {}}, safe_dict)
        
        output = {
            "expression": expression,
            "result": float(result),
            "success": True
        }
        
        logger.info(f"Result: {expression} = {result}")
        return output
        
    except Exception as e:
        logger.error(f"Calculation error: {e}")
        return {
            "expression": expression,
            "error": str(e),
            "success": False
        }


def is_smiles(text: str) -> bool:
    """Check if text is a valid SMILES string"""
    import re
    smiles_pattern = r'^[A-Za-z0-9@+\-\[\]\(\)=#$:/\\\.%]+$'
    return bool(re.match(smiles_pattern, text))


def is_multiple_smiles(text: str) -> bool:
    """Check if text contains multiple SMILES strings (separated by periods)"""
    # 检查是否有多个分子（通常用.分隔）
    parts = text.split('.')
    # 确保不是分子内部的点（如芳香环）
    return len(parts) > 1 and all(len(p.strip()) > 0 for p in parts)


def split_smiles(text: str) -> List[str]:
    """Split multiple SMILES strings separated by periods"""
    return [s.strip() for s in text.split('.') if s.strip()]


async def patent_check(smiles: str) -> Dict[str, Any]:
    """
    Check if molecule(s) are patented by querying SureChEMBL database.
    
    Args:
        smiles: SMILES string(s). Can be a single SMILES or multiple SMILES separated by periods.
        
    Returns:
        Dictionary containing:
            - success: Boolean indicating if check was successful
            - results: Dictionary mapping SMILES to patent status ("Patented" or "Novel")
            - error: Error message if check failed
            - message: Human-readable summary
    """
    logger.info(f"PatentCheck: Checking patent status for: {smiles}")
    
    # Validate input
    if not smiles or not isinstance(smiles, str):
        return {
            "success": False,
            "error": "Invalid input: SMILES string is required"
        }
    
    # Parse input SMILES
    if is_multiple_smiles(smiles):
        smiles_list = split_smiles(smiles)
        logger.info(f"Detected {len(smiles_list)} SMILES strings")
    else:
        smiles_list = [smiles]
    
    try:
        results = {}
        
        for smi in smiles_list:
            logger.debug(f"Checking patent status for: {smi}")
            
            try:
                # Query SureChEMBL database via molbloom
                is_patented = molbloom.buy(smi, canonicalize=True, catalog="surechembl")
                
                if is_patented:
                    results[smi] = "Patented"
                    logger.info(f"{smi} is patented")
                else:
                    results[smi] = "Novel"
                    logger.info(f"{smi} is novel (not patented)")
                    
            except Exception as e:
                logger.warning(f"Error checking {smi}: {e}")
                results[smi] = f"Error: {str(e)}"
        
        # Create summary message
        patented_count = sum(1 for v in results.values() if v == "Patented")
        novel_count = sum(1 for v in results.values() if v == "Novel")
        error_count = sum(1 for v in results.values() if v.startswith("Error"))
        
        summary_parts = []
        if patented_count > 0:
            summary_parts.append(f"{patented_count} patented")
        if novel_count > 0:
            summary_parts.append(f"{novel_count} novel")
        if error_count > 0:
            summary_parts.append(f"{error_count} errors")
        
        message = f"Patent check completed: {', '.join(summary_parts)}"
        
        return {
            "success": True,
            "results": results,
            "message": message,
            "summary": {
                "total": len(results),
                "patented": patented_count,
                "novel": novel_count,
                "errors": error_count
            }
        }
        
    except Exception as e:
        error_msg = f"Failed to check patent status: {str(e)}"
        logger.error(f"PatentCheck failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": error_msg
        }

