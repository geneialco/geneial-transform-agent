"""
UMLS API Client for integrating with the UMLS server.
This module provides functionality to search medical terminology, CUIs, and relationships.
"""

import os
import json
import logging
import requests
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from urllib.parse import urljoin, urlencode

logger = logging.getLogger(__name__)


@dataclass
class UMLSSearchResult:
    """Data class for UMLS search results."""

    code: str
    term: str
    description: Optional[str] = None
    ontology: Optional[str] = None


@dataclass
class CUIInfo:
    """Data class for CUI information."""

    cui: str
    name: str
    description: Optional[str] = None
    semantic_types: Optional[List[str]] = None


class UMLSClient:
    """Client for interacting with the UMLS API server."""

    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the UMLS client.

        Args:
            base_url: Base URL for the UMLS API server. Defaults to localhost:8000
        """
        self.base_url = base_url or os.getenv("UMLS_API_URL", "http://localhost:8000")
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

    def _make_request(
        self, endpoint: str, params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Make a request to the UMLS API.

        Args:
            endpoint: API endpoint to call
            params: Query parameters

        Returns:
            JSON response from the API

        Raises:
            requests.RequestException: If the request fails
        """
        url = urljoin(self.base_url, endpoint)

        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            # Only log as error if it's not a 404 (not found is expected for invalid terms)
            if hasattr(e, "response") and e.response.status_code == 404:
                logger.debug(f"UMLS API request returned 404 (not found): {e}")
            else:
                logger.error(f"UMLS API request failed: {e}")
            raise

    def search_terms(
        self, search_term: str, ontology: str = "HPO", limit: int = 10
    ) -> List[UMLSSearchResult]:
        """
        Search for medical terms in the specified ontology.

        Args:
            search_term: Term to search for
            ontology: Ontology to search in (HPO, SNOMEDCT_US, etc.)
            limit: Maximum number of results to return

        Returns:
            List of search results
        """
        params = {"search": search_term, "ontology": ontology, "limit": limit}

        try:
            response = self._make_request("/terms", params)
            results = []

            for item in response.get("results", []):
                result = UMLSSearchResult(
                    code=item.get("code", ""),
                    term=item.get("term", ""),
                    description=item.get("description"),
                    ontology=ontology,
                )
                results.append(result)

            return results
        except Exception as e:
            logger.error(f"Error searching terms: {e}")
            return []

    def search_cuis(self, query: str, limit: int = 10) -> List[str]:
        """
        Search for CUIs matching a given term.

        Args:
            query: Term to search for
            limit: Maximum number of results to return

        Returns:
            List of CUI identifiers
        """
        params = {"query": query, "limit": limit}

        try:
            response = self._make_request("/cuis", params)
            cuis = response.get("cuis", [])
            # Parse CUIs - they may come as dict objects with 'cui' key
            result = []
            for cui_item in cuis:
                if isinstance(cui_item, dict) and "cui" in cui_item:
                    result.append(cui_item["cui"])
                elif isinstance(cui_item, str):
                    result.append(cui_item)
                else:
                    result.append(str(cui_item))
            return result
        except Exception as e:
            logger.error(f"Error searching CUIs: {e}")
            return []

    def get_cui_info(self, cui: str) -> Optional[CUIInfo]:
        """
        Get detailed information about a specific CUI.

        Args:
            cui: CUI identifier

        Returns:
            CUI information or None if not found
        """
        try:
            response = self._make_request(f"/cuis/{cui}")
            return CUIInfo(
                cui=cui,
                name=response.get("name", ""),
                description=response.get("description"),
                semantic_types=response.get("semantic_types", []),
            )
        except Exception as e:
            logger.error(f"Error getting CUI info for {cui}: {e}")
            return None

    def get_cui_ancestors(self, cui: str) -> List[str]:
        """
        Get all ancestor CUIs for a given CUI.

        Args:
            cui: CUI identifier

        Returns:
            List of ancestor CUI identifiers
        """
        try:
            response = self._make_request(f"/cuis/{cui}/ancestors")
            return response.get("ancestors", [])
        except Exception as e:
            logger.error(f"Error getting ancestors for CUI {cui}: {e}")
            return []

    def get_cui_depth(self, cui: str) -> Optional[int]:
        """
        Get the depth of a CUI in the hierarchical structure.

        Args:
            cui: CUI identifier

        Returns:
            Depth value or None if not found
        """
        try:
            response = self._make_request(f"/cuis/{cui}/depth")
            return response.get("depth")
        except Exception as e:
            logger.error(f"Error getting depth for CUI {cui}: {e}")
            return None

    def find_lowest_common_ancestor(self, cui1: str, cui2: str) -> Optional[str]:
        """
        Find the lowest common ancestor (LCA) of two CUIs.

        Args:
            cui1: First CUI identifier
            cui2: Second CUI identifier

        Returns:
            LCA CUI identifier or None if not found
        """
        try:
            response = self._make_request(f"/cuis/{cui1}/{cui2}/lca")
            return response.get("lca")
        except Exception as e:
            logger.error(f"Error finding LCA for CUIs {cui1} and {cui2}: {e}")
            return None

    def calculate_wu_palmer_similarity(self, cui1: str, cui2: str) -> Optional[float]:
        """
        Calculate Wu-Palmer similarity between two CUIs.

        Args:
            cui1: First CUI identifier
            cui2: Second CUI identifier

        Returns:
            Similarity score (0.0-1.0) or None if calculation fails
        """
        try:
            response = self._make_request(f"/cuis/{cui1}/{cui2}/similarity/wu-palmer")
            return response.get("similarity")
        except Exception as e:
            logger.error(
                f"Error calculating similarity for CUIs {cui1} and {cui2}: {e}"
            )
            return None

    def get_hpo_term(self, cui: str) -> Optional[Dict[str, str]]:
        """
        Get HPO term and code from a CUI.

        Args:
            cui: CUI identifier

        Returns:
            Dictionary with HPO term info or None if not found
        """
        try:
            response = self._make_request(f"/cuis/{cui}/hpo")
            return response
        except Exception as e:
            logger.error(f"Error getting HPO term for CUI {cui}: {e}")
            return None

    def validate_terminology(
        self, terms: List[str], ontology: str = "HPO"
    ) -> Dict[str, bool]:
        """
        Validate a list of terms against an ontology.

        Args:
            terms: List of terms to validate
            ontology: Ontology to validate against

        Returns:
            Dictionary mapping terms to validation status
        """
        validation_results = {}

        for term in terms:
            try:
                results = self.search_terms(term, ontology, limit=1)
                validation_results[term] = len(results) > 0
            except Exception as e:
                logger.error(f"Error validating term {term}: {e}")
                validation_results[term] = False

        return validation_results

    def enhance_phenotype_data(self, phenotype_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance phenotype data with UMLS terminology information.

        Args:
            phenotype_data: Original phenotype data

        Returns:
            Enhanced phenotype data with UMLS annotations
        """
        enhanced_data = phenotype_data.copy()

        # Look for phenotype features to enhance
        if "phenotypicFeatures" in enhanced_data:
            for feature in enhanced_data["phenotypicFeatures"]:
                if "type" in feature and "label" in feature["type"]:
                    label = feature["type"]["label"]

                    # Search for matching terms in HPO
                    search_results = self.search_terms(label, "HPO", limit=1)
                    if search_results:
                        result = search_results[0]
                        # Add UMLS annotations
                        feature["type"]["umls_annotations"] = {
                            "code": result.code,
                            "term": result.term,
                            "description": result.description,
                            "ontology": result.ontology,
                        }

        return enhanced_data

    def health_check(self) -> bool:
        """
        Check if the UMLS server is accessible.

        Returns:
            True if server is accessible, False otherwise
        """
        try:
            # Use the /terms endpoint with minimal parameters for health check
            response = self.session.get(
                urljoin(self.base_url, "/terms"),
                params={"search": "test", "limit": 1},
                timeout=5,
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"UMLS server health check failed: {e}")
            return False


# Global client instance
_umls_client = None


def get_umls_client() -> UMLSClient:
    """
    Get a singleton instance of the UMLS client.

    Returns:
        UMLSClient instance
    """
    global _umls_client
    if _umls_client is None:
        _umls_client = UMLSClient()
    return _umls_client


def search_medical_terms(
    search_term: str, ontology: str = "HPO", limit: int = 10
) -> List[UMLSSearchResult]:
    """
    Convenience function to search medical terms using the global client.

    Args:
        search_term: Term to search for
        ontology: Ontology to search in
        limit: Maximum number of results

    Returns:
        List of search results
    """
    client = get_umls_client()
    return client.search_terms(search_term, ontology, limit)


def validate_medical_terminology(
    terms: List[str], ontology: str = "HPO"
) -> Dict[str, bool]:
    """
    Convenience function to validate medical terminology using the global client.

    Args:
        terms: List of terms to validate
        ontology: Ontology to validate against

    Returns:
        Dictionary mapping terms to validation status
    """
    client = get_umls_client()
    return client.validate_terminology(terms, ontology)
