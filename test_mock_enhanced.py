import unittest
import json
import os
import sys

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from introspection.atomspace_client import AtomSpaceClient
from mock import create_mock_cognitive_state, AtomSpaceMockGenerator

class TestMockAtomSpace(unittest.TestCase):
    def test_mock_cognitive_state(self):
        """Test the mock cognitive state generation."""
        client = AtomSpaceClient("http://localhost:8080/api/v1")  # Dummy endpoint
        mock_data = client.mock_get_cognitive_state()
        
        # Check that the mock data has the expected structure
        self.assertIn('atom_count', mock_data)
        self.assertIn('active_goals', mock_data)
        self.assertIn('attention_distribution', mock_data)
        self.assertIn('atom_distribution', mock_data)
        self.assertIn('cognitive_schematics', mock_data)
        self.assertIn('bottlenecks', mock_data)
        self.assertIn('recommendations', mock_data)
        
        # Check that there are some active goals
        self.assertTrue(len(mock_data['active_goals']) > 0)
        
        # Check that there are some bottlenecks
        self.assertTrue(len(mock_data['bottlenecks']) > 0)
        
        # Check that there are some recommendations
        self.assertTrue(len(mock_data['recommendations']) > 0)
    
    def test_modular_mock_generation(self):
        """Test the new modular mock data generation system."""
        # Test different complexity levels
        for complexity in ['simple', 'medium', 'complex']:
            mock_data = create_mock_cognitive_state(
                complexity_level=complexity,
                cognitive_focus='general'
            )
            
            self.assertIn('complexity_level', mock_data)
            self.assertEqual(mock_data['complexity_level'], complexity)
            self.assertIn('metadata', mock_data)
            self.assertIn('adaptive_patterns', mock_data['metadata'])
    
    def test_edge_case_cognitive_states(self):
        """Test edge cases and error handling for cognitive state generation."""
        # Test with extreme parameters
        generator = AtomSpaceMockGenerator(
            complexity_level='complex',
            cognitive_focus='attention'
        )
        
        # Test multiple generations for consistency
        for i in range(5):
            mock_data = generator.generate_cognitive_state()
            
            # Validate core structure
            self.assertIsInstance(mock_data['atom_count'], int)
            self.assertGreater(mock_data['atom_count'], 0)
            
            # Validate attention distribution
            attention = mock_data['attention_distribution']
            self.assertIn('attention_stats', attention)
            self.assertIn('attention_waves', attention)
            
            # Validate emergent properties
            emergent = mock_data['metadata']['emergent_properties']
            self.assertIn('consciousness_indicators', emergent)
            self.assertIn('distributed_cognition_sync', emergent)
    
    def test_error_handling_scenarios(self):
        """Test various error handling scenarios."""
        # Test with invalid parameters (should not crash)
        try:
            generator = AtomSpaceMockGenerator(
                complexity_level='invalid',
                cognitive_focus='unknown'
            )
            mock_data = generator.generate_cognitive_state()
            # Should fall back to defaults
            self.assertIsNotNone(mock_data)
        except Exception as e:
            self.fail(f"Mock generation should handle invalid parameters gracefully: {e}")

if __name__ == '__main__':
    unittest.main()