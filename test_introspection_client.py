import unittest
import os
import sys
import json
import requests
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from introspection.atomspace_client import AtomSpaceClient

class TestAtomSpaceClient(unittest.TestCase):
    def setUp(self):
        self.client = AtomSpaceClient("http://localhost:8080/api/v1")
    
    @patch('requests.Session.get')
    def test_test_connection(self, mock_get):
        # Mock the response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        # Test the method
        result = self.client.test_connection()
        
        # Verify the result
        self.assertTrue(result)
        mock_get.assert_called_once_with(
            "http://localhost:8080/api/v1/status",
            headers=self.client.headers,
            timeout=self.client.timeout
        )
    
    @patch('requests.Session.get')
    def test_get_atom_count(self, mock_get):
        # Mock the response
        mock_response = MagicMock()
        mock_response.json.return_value = {"count": 42}
        mock_get.return_value = mock_response
        
        # Test the method
        result = self.client.get_atom_count()
        
        # Verify the result
        self.assertEqual(result, 42)
        mock_get.assert_called_once()
    
    def test_mock_get_cognitive_state(self):
        # Test the mock data generation
        mock_data = self.client.mock_get_cognitive_state()
        
        # Check that the mock data has the expected structure
        self.assertIn('atom_count', mock_data)
        self.assertIn('active_goals', mock_data)
        self.assertIn('attention_distribution', mock_data)
        self.assertIn('atom_distribution', mock_data)
        self.assertIn('cognitive_schematics', mock_data)
        
        # Verify some values
        self.assertGreater(mock_data['atom_count'], 0)
        self.assertGreater(len(mock_data['active_goals']), 0)
        self.assertGreater(
            mock_data['attention_distribution']['high_sti_count'], 0
        )

if __name__ == '__main__':
    unittest.main()