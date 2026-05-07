# tools/modify_mol.py

import logging
from typing import Dict, Any, List, Optional
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import random

logger = logging.getLogger(__name__)


class MoleculeModifier:
    """Class to perform various molecular modifications"""
    
    def __init__(self):
        # Common functional groups for substitution
        self.functional_groups = [
            'C',      # Methyl
            'CC',     # Ethyl
            'C(C)C',  # Isopropyl
            'F',      # Fluoro
            'Cl',     # Chloro
            'Br',     # Bromo
            'O',      # Hydroxy
            'N',      # Amino
            'C(=O)O', # Carboxy
            'C(=O)C', # Acetyl
            'OC',     # Methoxy
            'C#N',    # Cyano
            'C(F)(F)F', # Trifluoromethyl
        ]
        
    def add_functional_group(self, mol: Chem.Mol, max_attempts: int = 10) -> Optional[Chem.Mol]:
        """Add a random functional group to the molecule"""
        for _ in range(max_attempts):
            try:
                # Get aromatic carbons or carbons with hydrogens
                pattern = Chem.MolFromSmarts('[c,C;H1,H2,H3]')
                matches = mol.GetSubstructMatches(pattern)
                
                if not matches:
                    continue
                
                # Select random position
                atom_idx = random.choice(matches)[0]
                
                # Select random functional group
                fg = random.choice(self.functional_groups)
                
                # Create modified molecule
                rw_mol = Chem.RWMol(mol)
                fg_mol = Chem.MolFromSmiles(fg)
                
                if fg_mol is None:
                    continue
                
                # Add functional group atoms
                new_atom_idx = rw_mol.AddAtom(fg_mol.GetAtomWithIdx(0))
                
                # Add bond
                rw_mol.AddBond(atom_idx, new_atom_idx, Chem.BondType.SINGLE)
                
                # Sanitize and return
                new_mol = rw_mol.GetMol()
                Chem.SanitizeMol(new_mol)
                
                return new_mol
                
            except Exception as e:
                logger.debug(f"Attempt failed: {e}")
                continue
        
        return None
    
    def remove_functional_group(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Remove a functional group from the molecule"""
        try:
            # Find terminal groups (atoms with only one heavy neighbor)
            candidates = []
            for atom in mol.GetAtoms():
                if atom.GetDegree() == 1 and atom.GetAtomicNum() != 1:  # Not hydrogen
                    candidates.append(atom.GetIdx())
            
            if not candidates:
                return None
            
            # Remove random terminal atom
            atom_to_remove = random.choice(candidates)
            rw_mol = Chem.RWMol(mol)
            rw_mol.RemoveAtom(atom_to_remove)
            
            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            
            return new_mol
            
        except Exception as e:
            logger.debug(f"Removal failed: {e}")
            return None
    
    def grow_carbon_chain(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Extend a carbon chain"""
        try:
            # Find carbons with hydrogens
            pattern = Chem.MolFromSmarts('[C;H1,H2,H3]')
            matches = mol.GetSubstructMatches(pattern)
            
            if not matches:
                return None
            
            atom_idx = random.choice(matches)[0]
            
            rw_mol = Chem.RWMol(mol)
            
            # Add CH2 group
            new_carbon = rw_mol.AddAtom(Chem.Atom(6))
            rw_mol.AddBond(atom_idx, new_carbon, Chem.BondType.SINGLE)
            
            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            
            return new_mol
            
        except Exception as e:
            logger.debug(f"Chain growth failed: {e}")
            return None
    
    def add_ring(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Add a small ring (cyclopropyl) to the molecule"""
        try:
            # Find carbons with at least 2 hydrogens
            pattern = Chem.MolFromSmarts('[C;H2,H3]')
            matches = mol.GetSubstructMatches(pattern)
            
            if not matches:
                return None
            
            atom_idx = random.choice(matches)[0]
            
            rw_mol = Chem.RWMol(mol)
            
            # Add cyclopropyl group: C1CC1
            c1 = rw_mol.AddAtom(Chem.Atom(6))
            c2 = rw_mol.AddAtom(Chem.Atom(6))
            c3 = rw_mol.AddAtom(Chem.Atom(6))
            
            rw_mol.AddBond(atom_idx, c1, Chem.BondType.SINGLE)
            rw_mol.AddBond(c1, c2, Chem.BondType.SINGLE)
            rw_mol.AddBond(c2, c3, Chem.BondType.SINGLE)
            rw_mol.AddBond(c3, c1, Chem.BondType.SINGLE)
            
            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            
            return new_mol
            
        except Exception as e:
            logger.debug(f"Ring addition failed: {e}")
            return None
    
    def substitute_atom(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """Substitute an atom with another atom"""
        try:
            # Define possible substitutions
            substitutions = {
                6: [7, 8],    # C -> N, O
                7: [6, 8],    # N -> C, O
                8: [6, 7],    # O -> C, N
                9: [17, 35],  # F -> Cl, Br
            }
            
            # Find atoms that can be substituted
            candidates = []
            for atom in mol.GetAtoms():
                if atom.GetAtomicNum() in substitutions:
                    candidates.append(atom.GetIdx())
            
            if not candidates:
                return None
            
            atom_idx = random.choice(candidates)
            old_atom_num = mol.GetAtomWithIdx(atom_idx).GetAtomicNum()
            new_atom_num = random.choice(substitutions[old_atom_num])
            
            rw_mol = Chem.RWMol(mol)
            rw_mol.GetAtomWithIdx(atom_idx).SetAtomicNum(new_atom_num)
            
            new_mol = rw_mol.GetMol()
            Chem.SanitizeMol(new_mol)
            
            return new_mol
            
        except Exception as e:
            logger.debug(f"Substitution failed: {e}")
            return None


def is_valid_molecule(mol: Chem.Mol, min_mw: float = 50, max_mw: float = 1000) -> bool:
    """Check if molecule meets basic validity criteria"""
    if mol is None:
        return False
    
    try:
        # Check molecular weight
        mw = Descriptors.MolWt(mol)
        if mw < min_mw or mw > max_mw:
            return False
        
        # Check if molecule can be sanitized
        Chem.SanitizeMol(mol)
        
        # Check if SMILES can be generated
        smiles = Chem.MolToSmiles(mol)
        if not smiles:
            return False
        
        return True
        
    except Exception as e:
        logger.debug(f"Validation failed: {e}")
        return False


async def modify_mol(
    smiles: str,
    num_variants: int = 1,
    modification_types: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Generate modified variants of a molecule from its SMILES string.
    
    Args:
        smiles: Input SMILES string
        num_variants: Number of variants to generate (default: 1)
        modification_types: List of modification types to use. 
                          Options: ['add_group', 'remove_group', 'grow_chain', 'add_ring', 'substitute']
                          If None, uses all types randomly.
        
    Returns:
        Dictionary containing:
            - success: Boolean indicating if modification was successful
            - original_smiles: The input SMILES
            - variants: List of modified SMILES strings
            - modifications: List of what modifications were applied
            - error: Error message if modification failed
    """
    logger.info(f"ModifyMol: Modifying molecule: {smiles}")
    
    # Validate input
    if not smiles or not isinstance(smiles, str):
        return {
            "success": False,
            "error": "Invalid input: SMILES string is required"
        }
    
    try:
        # Parse input molecule
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {
                "success": False,
                "error": f"Invalid SMILES string: {smiles}"
            }
        
        # Initialize modifier
        modifier = MoleculeModifier()
        
        # Define available modifications
        all_modifications = {
            'add_group': modifier.add_functional_group,
            'remove_group': modifier.remove_functional_group,
            'grow_chain': modifier.grow_carbon_chain,
            'add_ring': modifier.add_ring,
            'substitute': modifier.substitute_atom,
        }
        
        # Select modification types
        if modification_types is None:
            modification_types = list(all_modifications.keys())
        
        # Generate variants
        variants = []
        modifications_applied = []
        attempts = 0
        max_attempts = num_variants * 20  # Allow multiple attempts per variant
        
        while len(variants) < num_variants and attempts < max_attempts:
            attempts += 1
            
            # Select random modification
            mod_type = random.choice(modification_types)
            mod_func = all_modifications[mod_type]
            
            # Apply modification
            new_mol = mod_func(mol)
            
            # Validate and add if successful
            if is_valid_molecule(new_mol):
                new_smiles = Chem.MolToSmiles(new_mol)
                
                # Avoid duplicates
                if new_smiles not in variants and new_smiles != smiles:
                    variants.append(new_smiles)
                    modifications_applied.append(mod_type)
                    logger.info(f"Generated variant {len(variants)}: {new_smiles} ({mod_type})")
        
        if not variants:
            return {
                "success": False,
                "error": f"Failed to generate valid variants after {attempts} attempts"
            }
        
        # Return first variant if only one requested (matching the example output)
        if num_variants == 1:
            result_smiles = variants[0]
        else:
            result_smiles = variants
        
        return {
            "success": True,
            "original_smiles": smiles,
            "modified_smiles": result_smiles,
            "variants": variants,
            "modifications": modifications_applied,
            "message": f"Successfully generated {len(variants)} variant(s) from {smiles}"
        }
        
    except Exception as e:
        error_msg = f"Failed to modify molecule: {str(e)}"
        logger.error(f"ModifyMol failed: {e}", exc_info=True)
        return {
            "success": False,
            "error": error_msg
        }


# Tool definition for agent
MODIFY_MOL_TOOL = {
    "type": "function",
    "function": {
        "name": "modify_mol",
        "description": (
            "Generate modified variants of a molecule from its SMILES string. "
            "This tool performs structural modifications including: "
            "adding/removing functional groups, growing carbon chains, adding rings, and substituting atoms. "
            "Useful for exploring chemical space and generating novel molecular structures. "
            "Returns chemically valid SMILES strings that differ from the input."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "smiles": {
                    "type": "string",
                    "description": (
                        "Input SMILES string to modify. "
                        "Example: 'Oc1ccccc1' (phenol), 'CCO' (ethanol), 'c1ccccc1' (benzene)"
                    )
                },
                "num_variants": {
                    "type": "integer",
                    "description": "Number of modified variants to generate. Default is 1.",
                    "default": 1
                },
                "modification_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["add_group", "remove_group", "grow_chain", "add_ring", "substitute"]
                    },
                    "description": (
                        "Types of modifications to apply. Options: "
                        "'add_group' (add functional groups), 'remove_group' (remove groups), "
                        "'grow_chain' (extend carbon chains), 'add_ring' (add cyclopropyl), "
                        "'substitute' (replace atoms). If not specified, uses all types."
                    )
                }
            },
            "required": ["smiles"]
        }
    }
}
