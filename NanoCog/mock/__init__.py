"""
NanoCog Mock Data Generation Module

This module provides reusable mock data generation for AtomSpace cognitive states,
following principles of emergent cognition and neural-symbolic integration.

Key Features:
- Cognitive state simulation with realistic patterns
- Attention distribution modeling
- Goal-oriented schema generation
- Bottleneck detection simulation
- Adaptive recommendation generation

TODO: Future recursive expansions:
- Dynamic cognitive pattern evolution
- Multi-agent interaction simulation
- Temporal cognitive state transitions
- Adaptive learning curve modeling
"""

from .atomspace_mock import AtomSpaceMockGenerator
from .cognitive_patterns import CognitivePatternGenerator

__all__ = [
    'AtomSpaceMockGenerator',
    'CognitivePatternGenerator'
]

# Export convenience function for backward compatibility
def create_mock_cognitive_state(**kwargs):
    """
    Generate mock cognitive state data for testing or demonstration.
    
    Args:
        **kwargs: Configuration parameters for mock generation
        
    Returns:
        Dictionary with mock cognitive state data
    """
    generator = AtomSpaceMockGenerator(**kwargs)
    return generator.generate_cognitive_state()