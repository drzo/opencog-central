#!/usr/bin/env python3
"""
AtomSpace Client for NanoCog Introspection

This module provides a robust client for connecting to an AtomSpace via its REST API,
enabling introspective diagnostics of CogPrime-based systems. It includes methods for
querying various aspects of the system, pattern analysis, and cognitive state summarization.

Usage:
    client = AtomSpaceClient("http://localhost:8080/api/v1")
    if client.test_connection():
        summary = client.get_cognitive_state_summary()
        print(summary)
"""

import os
import time
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
from collections import defaultdict, Counter

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("nanocog.atomspace_client")

class AtomSpaceClient:
    """
    Client for connecting to and analyzing an AtomSpace via REST API.
    
    This client provides methods for querying the AtomSpace, analyzing patterns,
    and summarizing the cognitive state of a CogPrime-based system.
    """
    
    def __init__(self, 
                 endpoint: str, 
                 auth_token: Optional[str] = None,
                 timeout: int = 10,
                 max_retries: int = 3):
        """
        Initialize the AtomSpace client.
        
        Args:
            endpoint: The base URL of the AtomSpace REST API
            auth_token: Optional authentication token
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries for failed requests
        """
        self.endpoint = endpoint.rstrip('/')
        self.auth_token = auth_token
        self.timeout = timeout
        
        # Set up session with retry capability
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set up headers
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
    
    def test_connection(self) -> bool:
        """
        Test the connection to the AtomSpace.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = self.session.get(
                f"{self.endpoint}/status",
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            logger.info(f"Successfully connected to AtomSpace at {self.endpoint}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to connect to AtomSpace at {self.endpoint}: {str(e)}")
            return False
    
    def _make_request(self, 
                      method: str, 
                      path: str, 
                      params: Optional[Dict[str, Any]] = None,
                      data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the AtomSpace API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            path: API path relative to the endpoint
            params: Optional query parameters
            data: Optional request body data
            
        Returns:
            Response data as a dictionary
            
        Raises:
            requests.exceptions.RequestException: If the request fails
        """
        url = f"{self.endpoint}/{path.lstrip('/')}"
        
        try:
            if method.upper() == "GET":
                response = self.session.get(
                    url, 
                    headers=self.headers, 
                    params=params, 
                    timeout=self.timeout
                )
            elif method.upper() == "POST":
                response = self.session.post(
                    url, 
                    headers=self.headers, 
                    params=params, 
                    json=data, 
                    timeout=self.timeout
                )
            elif method.upper() == "PUT":
                response = self.session.put(
                    url, 
                    headers=self.headers, 
                    params=params, 
                    json=data, 
                    timeout=self.timeout
                )
            elif method.upper() == "DELETE":
                response = self.session.delete(
                    url, 
                    headers=self.headers, 
                    params=params, 
                    timeout=self.timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            
            # Try to parse as JSON, but handle non-JSON responses
            try:
                return response.json()
            except ValueError:
                return {"text": response.text}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to {url} failed: {str(e)}")
            raise
    
    # --- Basic AtomSpace Queries ---
    
    def get_atom_count(self) -> int:
        """
        Get the total number of atoms in the AtomSpace.
        
        Returns:
            Total atom count
        """
        try:
            response = self._make_request("GET", "atoms/count")
            return response.get("count", 0)
        except requests.exceptions.RequestException:
            logger.warning("Failed to get atom count, returning 0")
            return 0
    
    def get_atom_types(self) -> List[str]:
        """
        Get all atom types in the AtomSpace.
        
        Returns:
            List of atom type names
        """
        try:
            response = self._make_request("GET", "types")
            return response.get("types", [])
        except requests.exceptions.RequestException:
            logger.warning("Failed to get atom types, returning empty list")
            return []
    
    def get_atoms_by_type(self, 
                          atom_type: str, 
                          limit: int = 100, 
                          offset: int = 0) -> List[Dict[str, Any]]:
        """
        Get atoms of a specific type.
        
        Args:
            atom_type: The type of atoms to retrieve
            limit: Maximum number of atoms to retrieve
            offset: Offset for pagination
            
        Returns:
            List of atoms as dictionaries
        """
        try:
            response = self._make_request(
                "GET", 
                f"atoms/type/{atom_type}", 
                params={"limit": limit, "offset": offset}
            )
            return response.get("atoms", [])
        except requests.exceptions.RequestException:
            logger.warning(f"Failed to get atoms of type {atom_type}, returning empty list")
            return []
    
    def get_atom_by_handle(self, handle: str) -> Optional[Dict[str, Any]]:
        """
        Get an atom by its handle.
        
        Args:
            handle: The handle of the atom to retrieve
            
        Returns:
            Atom as a dictionary, or None if not found
        """
        try:
            response = self._make_request("GET", f"atoms/{handle}")
            return response
        except requests.exceptions.RequestException:
            logger.warning(f"Failed to get atom with handle {handle}")
            return None
    
    def get_incoming_set(self, handle: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the incoming set of an atom.
        
        Args:
            handle: The handle of the atom
            limit: Maximum number of atoms to retrieve
            
        Returns:
            List of atoms in the incoming set
        """
        try:
            response = self._make_request(
                "GET", 
                f"atoms/{handle}/incoming", 
                params={"limit": limit}
            )
            return response.get("atoms", [])
        except requests.exceptions.RequestException:
            logger.warning(f"Failed to get incoming set for atom {handle}, returning empty list")
            return []
    
    def get_outgoing_set(self, handle: str) -> List[Dict[str, Any]]:
        """
        Get the outgoing set of an atom.
        
        Args:
            handle: The handle of the atom
            
        Returns:
            List of atoms in the outgoing set
        """
        try:
            response = self._make_request("GET", f"atoms/{handle}/outgoing")
            return response.get("atoms", [])
        except requests.exceptions.RequestException:
            logger.warning(f"Failed to get outgoing set for atom {handle}, returning empty list")
            return []
    
    # --- Attention and STI/LTI Queries ---
    
    def get_atoms_by_sti(self, 
                         min_sti: float = 0.5, 
                         max_sti: Optional[float] = None, 
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get atoms with STI values in a specified range.
        
        Args:
            min_sti: Minimum STI value
            max_sti: Maximum STI value (optional)
            limit: Maximum number of atoms to retrieve
            
        Returns:
            List of atoms as dictionaries
        """
        params = {"min_sti": min_sti, "limit": limit}
        if max_sti is not None:
            params["max_sti"] = max_sti
            
        try:
            response = self._make_request("GET", "atoms/sti", params=params)
            return response.get("atoms", [])
        except requests.exceptions.RequestException:
            logger.warning(f"Failed to get atoms by STI, returning empty list")
            return []
    
    def get_atoms_by_lti(self, 
                         min_lti: float = 0.5, 
                         max_lti: Optional[float] = None, 
                         limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get atoms with LTI values in a specified range.
        
        Args:
            min_lti: Minimum LTI value
            max_lti: Maximum LTI value (optional)
            limit: Maximum number of atoms to retrieve
            
        Returns:
            List of atoms as dictionaries
        """
        params = {"min_lti": min_lti, "limit": limit}
        if max_lti is not None:
            params["max_lti"] = max_lti
            
        try:
            response = self._make_request("GET", "atoms/lti", params=params)
            return response.get("atoms", [])
        except requests.exceptions.RequestException:
            logger.warning(f"Failed to get atoms by LTI, returning empty list")
            return []
    
    def get_attention_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about attention allocation in the AtomSpace.
        
        Returns:
            Dictionary with attention statistics
        """
        try:
            response = self._make_request("GET", "attention/statistics")
            return response
        except requests.exceptions.RequestException:
            logger.warning("Failed to get attention statistics, returning empty dict")
            return {}
    
    # --- Goal System Queries ---
    
    def get_active_goals(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get currently active goals in the system.
        
        Args:
            limit: Maximum number of goals to retrieve
            
        Returns:
            List of active goals as dictionaries
        """
        try:
            response = self._make_request("GET", "goals/active", params={"limit": limit})
            return response.get("goals", [])
        except requests.exceptions.RequestException:
            logger.warning("Failed to get active goals, returning empty list")
            return []
    
    def get_goal_hierarchy(self, goal_handle: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the goal hierarchy, optionally starting from a specific goal.
        
        Args:
            goal_handle: Optional handle of the root goal
            
        Returns:
            Dictionary representing the goal hierarchy
        """
        try:
            params = {}
            if goal_handle:
                params["root"] = goal_handle
                
            response = self._make_request("GET", "goals/hierarchy", params=params)
            return response
        except requests.exceptions.RequestException:
            logger.warning("Failed to get goal hierarchy, returning empty dict")
            return {}
    
    # --- Pattern Mining and Analysis ---
    
    def get_frequent_patterns(self, 
                              min_support: float = 0.1, 
                              max_patterns: int = 20) -> List[Dict[str, Any]]:
        """
        Get frequent patterns in the AtomSpace.
        
        Args:
            min_support: Minimum support threshold for patterns
            max_patterns: Maximum number of patterns to retrieve
            
        Returns:
            List of patterns as dictionaries
        """
        try:
            response = self._make_request(
                "GET", 
                "patterns/frequent", 
                params={"min_support": min_support, "max_patterns": max_patterns}
            )
            return response.get("patterns", [])
        except requests.exceptions.RequestException:
            logger.warning("Failed to get frequent patterns, returning empty list")
            return []
    
    def get_surprising_patterns(self, 
                                max_patterns: int = 20) -> List[Dict[str, Any]]:
        """
        Get surprising (low probability but significant) patterns in the AtomSpace.
        
        Args:
            max_patterns: Maximum number of patterns to retrieve
            
        Returns:
            List of patterns as dictionaries
        """
        try:
            response = self._make_request(
                "GET", 
                "patterns/surprising", 
                params={"max_patterns": max_patterns}
            )
            return response.get("patterns", [])
        except requests.exceptions.RequestException:
            logger.warning("Failed to get surprising patterns, returning empty list")
            return []
    
    def analyze_atom_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of atoms by type in the AtomSpace.
        
        Returns:
            Dictionary with distribution statistics
        """
        try:
            # Get all atom types
            atom_types = self.get_atom_types()
            
            # Count atoms of each type
            type_counts = {}
            for atom_type in atom_types:
                try:
                    count_response = self._make_request(
                        "GET", 
                        f"atoms/type/{atom_type}/count"
                    )
                    type_counts[atom_type] = count_response.get("count", 0)
                except requests.exceptions.RequestException:
                    type_counts[atom_type] = 0
            
            # Calculate percentages
            total_atoms = sum(type_counts.values())
            type_percentages = {
                atom_type: (count / total_atoms * 100) if total_atoms > 0 else 0
                for atom_type, count in type_counts.items()
            }
            
            # Sort by count (descending)
            sorted_types = sorted(
                type_counts.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return {
                "total_atoms": total_atoms,
                "type_counts": type_counts,
                "type_percentages": type_percentages,
                "sorted_types": sorted_types
            }
        except Exception as e:
            logger.error(f"Failed to analyze atom distribution: {str(e)}")
            return {
                "total_atoms": 0,
                "type_counts": {},
                "type_percentages": {},
                "sorted_types": []
            }
    
    def analyze_attention_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of attention (STI/LTI) in the AtomSpace.
        
        Returns:
            Dictionary with attention distribution statistics
        """
        try:
            # Get attention statistics
            stats = self.get_attention_statistics()
            
            # Get high STI atoms for more detailed analysis
            high_sti_atoms = self.get_atoms_by_sti(min_sti=0.5, limit=200)
            
            # Count high STI atoms by type
            high_sti_by_type = defaultdict(int)
            for atom in high_sti_atoms:
                atom_type = atom.get("type", "unknown")
                high_sti_by_type[atom_type] += 1
            
            # Calculate STI distribution
            sti_ranges = {
                "very_high": (0.8, 1.0),
                "high": (0.6, 0.8),
                "medium": (0.4, 0.6),
                "low": (0.2, 0.4),
                "very_low": (0.0, 0.2)
            }
            
            sti_distribution = {}
            for range_name, (min_val, max_val) in sti_ranges.items():
                try:
                    atoms = self.get_atoms_by_sti(min_sti=min_val, max_sti=max_val, limit=1)
                    count_response = self._make_request(
                        "GET", 
                        "atoms/sti/count", 
                        params={"min_sti": min_val, "max_sti": max_val}
                    )
                    sti_distribution[range_name] = count_response.get("count", 0)
                except requests.exceptions.RequestException:
                    sti_distribution[range_name] = 0
            
            return {
                "attention_stats": stats,
                "high_sti_count": len(high_sti_atoms),
                "high_sti_by_type": dict(high_sti_by_type),
                "sti_distribution": sti_distribution
            }
        except Exception as e:
            logger.error(f"Failed to analyze attention distribution: {str(e)}")
            return {
                "attention_stats": {},
                "high_sti_count": 0,
                "high_sti_by_type": {},
                "sti_distribution": {}
            }
    
    def analyze_cognitive_schematics(self, limit: int = 100) -> Dict[str, Any]:
        """
        Analyze cognitive schematics (Context -> Procedure -> Goal) in the AtomSpace.
        
        Args:
            limit: Maximum number of schematics to analyze
            
        Returns:
            Dictionary with schematic analysis
        """
        try:
            # Get cognitive schematics
            schematics = self._make_request(
                "GET", 
                "schematics", 
                params={"limit": limit}
            ).get("schematics", [])
            
            # Analyze success rates
            success_counts = {"successful": 0, "failed": 0, "unknown": 0}
            
            for schematic in schematics:
                status = schematic.get("status", "unknown").lower()
                if status == "successful":
                    success_counts["successful"] += 1
                elif status == "failed":
                    success_counts["failed"] += 1
                else:
                    success_counts["unknown"] += 1
            
            # Calculate success rate
            total_known = success_counts["successful"] + success_counts["failed"]
            success_rate = (success_counts["successful"] / total_known * 100) if total_known > 0 else 0
            
            # Group by goal
            schematics_by_goal = defaultdict(list)
            for schematic in schematics:
                goal = schematic.get("goal", {}).get("name", "unknown")
                schematics_by_goal[goal].append(schematic)
            
            # Find most common goals
            goal_counts = {goal: len(schems) for goal, schems in schematics_by_goal.items()}
            sorted_goals = sorted(goal_counts.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "total_schematics": len(schematics),
                "success_counts": success_counts,
                "success_rate": success_rate,
                "schematics_by_goal": {k: len(v) for k, v in schematics_by_goal.items()},
                "top_goals": sorted_goals[:10]
            }
        except Exception as e:
            logger.error(f"Failed to analyze cognitive schematics: {str(e)}")
            return {
                "total_schematics": 0,
                "success_counts": {"successful": 0, "failed": 0, "unknown": 0},
                "success_rate": 0,
                "schematics_by_goal": {},
                "top_goals": []
            }
    
    # --- Comprehensive Analysis and Summaries ---
    
    def get_cognitive_state_summary(self) -> Dict[str, Any]:
        """
        Get a comprehensive summary of the cognitive state of the system.
        
        Returns:
            Dictionary with various aspects of the cognitive state
        """
        try:
            summary = {
                "timestamp": datetime.now().isoformat(),
                "atom_count": self.get_atom_count(),
                "active_goals": self.get_active_goals(limit=10),
                "attention_distribution": self.analyze_attention_distribution(),
                "atom_distribution": self.analyze_atom_distribution(),
                "cognitive_schematics": self.analyze_cognitive_schematics(limit=50)
            }
            
            # Add high-level metrics
            summary["metrics"] = {
                "goal_count": len(summary["active_goals"]),
                "high_sti_count": summary["attention_distribution"]["high_sti_count"],
                "schematic_success_rate": summary["cognitive_schematics"]["success_rate"],
                "atom_type_diversity": len(summary["atom_distribution"]["type_counts"])
            }
            
            return summary
        except Exception as e:
            logger.error(f"Failed to get cognitive state summary: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "atom_count": 0,
                "active_goals": [],
                "attention_distribution": {},
                "atom_distribution": {},
                "cognitive_schematics": {},
                "metrics": {}
            }
    
    def detect_cognitive_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Detect potential bottlenecks in the cognitive system.
        
        Returns:
            List of detected bottlenecks with descriptions and severity
        """
        bottlenecks = []
        
        try:
            # Get cognitive state summary
            summary = self.get_cognitive_state_summary()
            
            # Check for attention allocation issues
            attention_dist = summary.get("attention_distribution", {})
            sti_dist = attention_dist.get("sti_distribution", {})
            
            if sti_dist.get("very_high", 0) > 100:
                bottlenecks.append({
                    "type": "attention_concentration",
                    "description": "Too many atoms with very high STI values",
                    "severity": "medium",
                    "count": sti_dist.get("very_high", 0),
                    "recommendation": "Consider increasing ECAN decay rate or adjusting STI spread factors"
                })
            
            # Check for goal proliferation
            goal_count = len(summary.get("active_goals", []))
            if goal_count > 7:
                bottlenecks.append({
                    "type": "goal_proliferation",
                    "description": "Excessive number of active goals",
                    "severity": "high" if goal_count > 15 else "medium",
                    "count": goal_count,
                    "recommendation": "Increase goal selection threshold or implement stricter goal pruning"
                })
            
            # Check for cognitive schematic issues
            schematics = summary.get("cognitive_schematics", {})
            success_rate = schematics.get("success_rate", 0)
            
            if success_rate < 40:
                bottlenecks.append({
                    "type": "low_schematic_success",
                    "description": "Low success rate for cognitive schematics",
                    "severity": "high" if success_rate < 20 else "medium",
                    "rate": success_rate,
                    "recommendation": "Review procedure learning parameters or context definitions"
                })
            
            # Check for atom type imbalance
            atom_dist = summary.get("atom_distribution", {})
            type_percentages = atom_dist.get("type_percentages", {})
            
            # Look for dominant atom types (>50% of one type)
            for atom_type, percentage in type_percentages.items():
                if percentage > 50:
                    bottlenecks.append({
                        "type": "atom_type_imbalance",
                        "description": f"Excessive dominance of {atom_type} atoms",
                        "severity": "medium",
                        "percentage": percentage,
                        "recommendation": "Diversify knowledge representation or adjust creation parameters"
                    })
            
            return bottlenecks
        except Exception as e:
            logger.error(f"Failed to detect cognitive bottlenecks: {str(e)}")
            return [{
                "type": "analysis_error",
                "description": f"Error during bottleneck detection: {str(e)}",
                "severity": "unknown"
            }]
    
    def generate_introspection_report(self, 
                                     include_bottlenecks: bool = True,
                                     include_recommendations: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive introspection report.
        
        Args:
            include_bottlenecks: Whether to include bottleneck detection
            include_recommendations: Whether to include recommendations
            
        Returns:
            Dictionary with the complete report
        """
        try:
            # Get basic cognitive state
            summary = self.get_cognitive_state_summary()
            
            # Add bottlenecks if requested
            if include_bottlenecks:
                summary["bottlenecks"] = self.detect_cognitive_bottlenecks()
            
            # Add recommendations if requested
            if include_recommendations and include_bottlenecks:
                recommendations = []
                for bottleneck in summary.get("bottlenecks", []):
                    if "recommendation" in bottleneck:
                        recommendations.append({
                            "for_issue": bottleneck["type"],
                            "severity": bottleneck["severity"],
                            "action": bottleneck["recommendation"]
                        })
                
                summary["recommendations"] = recommendations
            
            # Add a human-readable summary
            readable_summary = self._generate_readable_summary(summary)
            summary["readable_summary"] = readable_summary
            
            return summary
        except Exception as e:
            logger.error(f"Failed to generate introspection report: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "error": str(e),
                "readable_summary": "Failed to generate introspection report due to an error."
            }
    
    def _generate_readable_summary(self, summary: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary from the introspection data.
        
        Args:
            summary: The introspection summary dictionary
            
        Returns:
            A human-readable summary string
        """
        lines = []
        
        # Basic information
        lines.append(f"AtomSpace Introspection Report ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        lines.append("=" * 50)
        
        # Atom count
        lines.append(f"Total atoms: {summary.get('atom_count', 'unknown')}")
        
        # Active goals
        active_goals = summary.get("active_goals", [])
        lines.append(f"\nActive goals: {len(active_goals)}")
        for i, goal in enumerate(active_goals[:5]):  # Show top 5
            goal_name = goal.get("name", "Unnamed")
            goal_sti = goal.get("sti", 0.0)
            lines.append(f"  {i+1}. {goal_name} (STI: {goal_sti:.2f})")
        
        if len(active_goals) > 5:
            lines.append(f"  ... and {len(active_goals) - 5} more goals")
        
        # Attention distribution
        attention_dist = summary.get("attention_distribution", {})
        sti_dist = attention_dist.get("sti_distribution", {})
        
        if sti_dist:
            lines.append("\nSTI Distribution:")
            for range_name, count in sti_dist.items():
                lines.append(f"  {range_name}: {count} atoms")
        
        # Atom type distribution
        atom_dist = summary.get("atom_distribution", {})
        sorted_types = atom_dist.get("sorted_types", [])
        
        if sorted_types:
            lines.append("\nTop atom types:")
            for atom_type, count in sorted_types[:5]:  # Show top 5
                percentage = atom_dist.get("type_percentages", {}).get(atom_type, 0)
                lines.append(f"  {atom_type}: {count} atoms ({percentage:.1f}%)")
        
        # Cognitive schematics
        schematics = summary.get("cognitive_schematics", {})
        success_rate = schematics.get("success_rate", 0)
        
        lines.append(f"\nCognitive schematics: {schematics.get('total_schematics', 0)} total")
        lines.append(f"  Success rate: {success_rate:.1f}%")
        
        # Top goals with schematics
        top_goals = schematics.get("top_goals", [])
        if top_goals:
            lines.append("  Top goals with schematics:")
            for goal, count in top_goals[:3]:  # Show top 3
                lines.append(f"    {goal}: {count} schematics")
        
        # Bottlenecks
        bottlenecks = summary.get("bottlenecks", [])
        if bottlenecks:
            lines.append("\nDetected bottlenecks:")
            for bottleneck in bottlenecks:
                severity = bottleneck.get("severity", "unknown").upper()
                description = bottleneck.get("description", "Unknown issue")
                lines.append(f"  [{severity}] {description}")
        
        # Recommendations
        recommendations = summary.get("recommendations", [])
        if recommendations:
            lines.append("\nRecommendations:")
            for i, rec in enumerate(recommendations):
                lines.append(f"  {i+1}. {rec.get('action', 'No action specified')}")
        
        return "\n".join(lines)
    
    # --- Mock methods for testing without a real AtomSpace ---
    
    def mock_get_cognitive_state(self) -> Dict[str, Any]:
        """
        Generate mock cognitive state data for testing.
        
        Returns:
            Dictionary with mock cognitive state data
        """
        import random
        
        # Mock atom count
        atom_count = random.randint(1000, 10000)
        
        # Mock active goals
        active_goals = []
        for i in range(random.randint(3, 8)):
            active_goals.append({
                "handle": f"0x{random.randint(1000, 9999):x}",
                "name": f"Goal{i+1}",
                "type": "ConceptNode",
                "sti": random.uniform(0.5, 0.95),
                "lti": random.uniform(0.1, 0.5),
                "tv": [random.uniform(0.7, 0.99), random.uniform(0.6, 0.9)]
            })
        
        # Mock attention distribution
        attention_distribution = {
            "attention_stats": {
                "avg_sti": random.uniform(0.1, 0.3),
                "max_sti": random.uniform(0.8, 0.99),
                "min_sti": random.uniform(0.01, 0.1),
                "std_dev_sti": random.uniform(0.1, 0.3),
                "avg_lti": random.uniform(0.1, 0.3),
                "max_lti": random.uniform(0.7, 0.9),
                "min_lti": random.uniform(0.01, 0.1)
            },
            "high_sti_count": random.randint(50, 200),
            "high_sti_by_type": {
                "ConceptNode": random.randint(20, 100),
                "PredicateNode": random.randint(10, 50),
                "ListLink": random.randint(5, 30),
                "EvaluationLink": random.randint(10, 40),
                "HebbianLink": random.randint(0, 20)
            },
            "sti_distribution": {
                "very_high": random.randint(10, 50),
                "high": random.randint(50, 150),
                "medium": random.randint(200, 500),
                "low": random.randint(500, 1000),
                "very_low": random.randint(1000, 5000)
            }
        }
        
        # Mock atom distribution
        atom_types = [
            "ConceptNode", "PredicateNode", "ListLink", "EvaluationLink", 
            "HebbianLink", "InheritanceLink", "SchemaNode", "VariableNode",
            "GroundedSchemaNode", "GroundedPredicateNode", "AndLink", "OrLink"
        ]
        
        type_counts = {}
        for atom_type in atom_types:
            type_counts[atom_type] = random.randint(50, 1000)
        
        total_atoms = sum(type_counts.values())
        type_percentages = {
            atom_type: (count / total_atoms * 100)
            for atom_type, count in type_counts.items()
        }
        
        sorted_types = sorted(
            type_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        atom_distribution = {
            "total_atoms": total_atoms,
            "type_counts": type_counts,
            "type_percentages": type_percentages,
            "sorted_types": sorted_types
        }
        
        # Mock cognitive schematics
        schematics = {
            "total_schematics": random.randint(50, 200),
            "success_counts": {
                "successful": random.randint(30, 150),
                "failed": random.randint(10, 50),
                "unknown": random.randint(0, 20)
            },
            "schematics_by_goal": {
                f"Goal{i}": random.randint(5, 30)
                for i in range(1, random.randint(5, 10))
            }
        }
        
        # Calculate success rate
        total_known = schematics["success_counts"]["successful"] + schematics["success_counts"]["failed"]
        success_rate = (schematics["success_counts"]["successful"] / total_known * 100) if total_known > 0 else 0
        schematics["success_rate"] = success_rate
        
        # Sort goals by schematic count
        sorted_goals = sorted(
            schematics["schematics_by_goal"].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        schematics["top_goals"] = sorted_goals
        
        # Generate bottlenecks based on the mock data
        bottlenecks = []
        
        # Check for attention concentration
        if attention_distribution["sti_distribution"]["very_high"] > 40:
            bottlenecks.append({
                "type": "attention_concentration",
                "description": "Too many atoms with very high STI values",
                "severity": "medium",
                "count": attention_distribution["sti_distribution"]["very_high"],
                "recommendation": "Consider increasing ECAN decay rate or adjusting STI spread factors"
            })
        
        # Check for goal proliferation
        if len(active_goals) > 7:
            bottlenecks.append({
                "type": "goal_proliferation",
                "description": "Excessive number of active goals",
                "severity": "high" if len(active_goals) > 15 else "medium",
                "count": len(active_goals),
                "recommendation": "Increase goal selection threshold or implement stricter goal pruning"
            })
        
        # Check for cognitive schematic issues
        if success_rate < 40:
            bottlenecks.append({
                "type": "low_schematic_success",
                "description": "Low success rate for cognitive schematics",
                "severity": "high" if success_rate < 20 else "medium",
                "rate": success_rate,
                "recommendation": "Review procedure learning parameters or context definitions"
            })
        
        # Check for atom type imbalance
        for atom_type, percentage in type_percentages.items():
            if percentage > 50:
                bottlenecks.append({
                    "type": "atom_type_imbalance",
                    "description": f"Excessive dominance of {atom_type} atoms",
                    "severity": "medium",
                    "percentage": percentage,
                    "recommendation": "Diversify knowledge representation or adjust creation parameters"
                })
        
        # Generate recommendations
        recommendations = []
        for bottleneck in bottlenecks:
            if "recommendation" in bottleneck:
                recommendations.append({
                    "for_issue": bottleneck["type"],
                    "severity": bottleneck["severity"],
                    "action": bottleneck["recommendation"]
                })
        
        # Assemble the complete mock data
        return {
            "timestamp": datetime.now().isoformat(),
            "atom_count": atom_count,
            "active_goals": active_goals,
            "attention_distribution": attention_distribution,
            "atom_distribution": atom_distribution,
            "cognitive_schematics": schematics,
            "bottlenecks": bottlenecks,
            "recommendations": recommendations,
            "metrics": {
                "goal_count": len(active_goals),
                "high_sti_count": attention_distribution["high_sti_count"],
                "schematic_success_rate": success_rate,
                "atom_type_diversity": len(atom_distribution["type_counts"])
            }
        }

# --- Command-line interface for testing ---

def main():
    """
    Command-line interface for testing the AtomSpaceClient.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="AtomSpace Client for NanoCog Introspection")
    parser.add_argument("--endpoint", type=str, help="AtomSpace REST API endpoint")
    parser.add_argument("--auth-token", type=str, help="Authentication token (if required)")
    parser.add_argument("--mock", action="store_true", help="Use mock data instead of connecting to an AtomSpace")
    parser.add_argument("--output", type=str, help="Output file for the report (JSON format)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.getLogger("nanocog.atomspace_client").setLevel(logging.DEBUG)
    
    if args.mock:
        print("Using mock data (no AtomSpace connection)")
        client = AtomSpaceClient("http://localhost:8080/api/v1")  # Dummy endpoint
        report = client.mock_get_cognitive_state()
    else:
        if not args.endpoint:
            print("Error: AtomSpace endpoint is required when not using mock data")
            parser.print_help()
            return
        
        print(f"Connecting to AtomSpace at {args.endpoint}...")
        client = AtomSpaceClient(args.endpoint, auth_token=args.auth_token)
        
        if not client.test_connection():
            print("Failed to connect to AtomSpace. Use --mock to generate mock data instead.")
            return
        
        print("Generating introspection report...")
        report = client.generate_introspection_report()
    
    # Print readable summary
    if "readable_summary" in report:
        print("\n" + report["readable_summary"])
    
    # Save to file if requested
    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to {args.output}")
        except Exception as e:
            print(f"Error saving report to {args.output}: {str(e)}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
