"""
UMLS tools for workflow agents.
"""

import logging
from typing import Annotated, List, Dict, Any
from langchain_core.tools import tool
from .decorators import log_io
from ..utils.umls_client import get_umls_client

logger = logging.getLogger(__name__)


@tool
@log_io
def search_medical_terms_tool(
    search_term: Annotated[str, "Medical term to search for"],
    ontology: Annotated[str, "Ontology to search in (HPO, SNOMEDCT_US, etc.)"] = "HPO",
    limit: Annotated[int, "Maximum number of results to return"] = 10,
) -> str:
    """Search for medical terms in UMLS ontologies (HPO, SNOMED CT, etc.)."""
    try:
        umls_client = get_umls_client()

        # Check if UMLS server is accessible
        if not umls_client.health_check():
            return "❌ UMLS server is not accessible. Please ensure the UMLS server is running."

        results = umls_client.search_terms(search_term, ontology, limit)

        if results:
            output = f"✅ Found {len(results)} medical terms for '{search_term}' in {ontology}:\n"
            for i, result in enumerate(results, 1):
                output += f"{i}. **{result.term}** ({result.code})"
                if result.description:
                    output += f"\n   Description: {result.description}"
                output += "\n"
            return output
        else:
            return f"ℹ️ No medical terms found for '{search_term}' in {ontology}"

    except Exception as e:
        error_msg = f"Error searching medical terms: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
@log_io
def validate_medical_terminology_tool(
    terms: Annotated[List[str], "List of medical terms to validate"],
    ontology: Annotated[str, "Ontology to validate against"] = "HPO",
) -> str:
    """Validate medical terms against UMLS ontologies."""
    try:
        umls_client = get_umls_client()

        # Check if UMLS server is accessible
        if not umls_client.health_check():
            return "❌ UMLS server is not accessible. Please ensure the UMLS server is running."

        validation_results = umls_client.validate_terminology(terms, ontology)

        valid_terms = [
            term for term, is_valid in validation_results.items() if is_valid
        ]
        invalid_terms = [
            term for term, is_valid in validation_results.items() if not is_valid
        ]

        output = f"📊 Validation results for {len(terms)} terms against {ontology}:\n"

        if valid_terms:
            output += f"✅ Valid terms ({len(valid_terms)}): {', '.join(valid_terms)}\n"

        if invalid_terms:
            output += (
                f"❌ Invalid terms ({len(invalid_terms)}): {', '.join(invalid_terms)}\n"
            )

        return output

    except Exception as e:
        error_msg = f"Error validating terminology: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
@log_io
def search_cuis_tool(
    query: Annotated[str, "Medical term to search for CUIs"],
    limit: Annotated[int, "Maximum number of CUIs to return"] = 10,
) -> str:
    """Search for CUIs (Concept Unique Identifiers) matching a medical term."""
    try:
        umls_client = get_umls_client()

        # Check if UMLS server is accessible
        if not umls_client.health_check():
            return "❌ UMLS server is not accessible. Please ensure the UMLS server is running."

        cuis = umls_client.search_cuis(query, limit)

        if cuis:
            output = f"✅ Found {len(cuis)} CUIs for '{query}':\n"
            output += ", ".join(cuis[:10])  # Show first 10
            return output
        else:
            return f"ℹ️ No CUIs found for '{query}'"

    except Exception as e:
        error_msg = f"Error searching CUIs: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
@log_io
def get_cui_info_tool(cui: Annotated[str, "CUI identifier (e.g., C0011900)"]) -> str:
    """Get detailed information about a specific CUI."""
    try:
        umls_client = get_umls_client()

        # Check if UMLS server is accessible
        if not umls_client.health_check():
            return "❌ UMLS server is not accessible. Please ensure the UMLS server is running."

        cui_info = umls_client.get_cui_info(cui)

        if cui_info:
            output = f"✅ CUI Information:\n**CUI:** {cui_info.cui}\n**Name:** {cui_info.name}"
            if cui_info.description:
                output += f"\n**Description:** {cui_info.description}"
            if cui_info.semantic_types:
                output += f"\n**Semantic Types:** {', '.join(cui_info.semantic_types)}"
            return output
        else:
            return f"❌ CUI '{cui}' not found"

    except Exception as e:
        error_msg = f"Error getting CUI info: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
@log_io
def calculate_cui_similarity_tool(
    cui1: Annotated[str, "First CUI identifier"],
    cui2: Annotated[str, "Second CUI identifier"],
) -> str:
    """Calculate Wu-Palmer similarity between two CUIs."""
    try:
        umls_client = get_umls_client()

        # Check if UMLS server is accessible
        if not umls_client.health_check():
            return "❌ UMLS server is not accessible. Please ensure the UMLS server is running."

        similarity = umls_client.calculate_wu_palmer_similarity(cui1, cui2)

        if similarity is not None:
            return (
                f"✅ Wu-Palmer similarity between {cui1} and {cui2}: {similarity:.4f}"
            )
        else:
            return f"❌ Could not calculate similarity between {cui1} and {cui2}"

    except Exception as e:
        error_msg = f"Error calculating similarity: {str(e)}"
        logger.error(error_msg)
        return error_msg


@tool
@log_io
def enhance_phenotype_data_tool(
    phenotype_data: Annotated[
        Dict[str, Any], "Phenotype data to enhance with UMLS annotations"
    ],
    ontology: Annotated[str, "Primary ontology for enhancement"] = "HPO",
) -> str:
    """Enhance phenotype data with UMLS terminology annotations."""
    try:
        umls_client = get_umls_client()

        # Check if UMLS server is accessible
        if not umls_client.health_check():
            return "❌ UMLS server is not accessible. Please ensure the UMLS server is running."

        enhanced_data = umls_client.enhance_phenotype_data(phenotype_data)

        return f"✅ Enhanced phenotype data with UMLS annotations:\n```json\n{enhanced_data}\n```"

    except Exception as e:
        error_msg = f"Error enhancing phenotype data: {str(e)}"
        logger.error(error_msg)
        return error_msg
