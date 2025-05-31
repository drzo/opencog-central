import unittest
import os
import sys
import json
from unittest.mock import patch, MagicMock

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Create a mock for the ModelConfig
class MockModelConfig:
    def __init__(self, model_path, device="cpu", max_tokens=2048):
        self.model_path = model_path
        self.device = device
        self.max_tokens = max_tokens
        self.model = None
        self.tokenizer = None
        self.model_info = {}
    
    def load_model(self):
        self.model = MagicMock()
        self.tokenizer = MagicMock()
        self.model_info = {
            "model_args": {"n_layer": 2, "n_head": 2, "n_embd": 64},
            "iter_num": 0,
            "best_val_loss": 999.0,
            "checkpoint_path": self.model_path
        }
        return True
    
    def generate(self, prompt, max_new_tokens=500, temperature=0.7, top_k=200, callback=None):
        if callback:
            callback("This ")
            callback("is ")
            callback("a ")
            callback("test ")
            callback("response.")
        return "This is a test response."

# Import with patching
with patch('nctalk.ModelConfig', MockModelConfig):
    from nctalk import ConversationHistory, DiagnosticMode

class TestConversationHistory(unittest.TestCase):
    def test_add_message(self):
        history = ConversationHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there")
        
        messages = history.get_messages()
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[0]["content"], "Hello")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(messages[1]["content"], "Hi there")
    
    def test_clear(self):
        history = ConversationHistory()
        history.add_message("user", "Hello")
        history.clear()
        
        messages = history.get_messages()
        self.assertEqual(len(messages), 0)
    
    def test_format_for_prompt(self):
        history = ConversationHistory()
        history.add_message("user", "Hello")
        history.add_message("assistant", "Hi there")
        
        prompt = history.format_for_prompt()
        self.assertIn("User: Hello", prompt)
        self.assertIn("NanoCog: Hi there", prompt)
        self.assertTrue(prompt.endswith("NanoCog: "))

class TestDiagnosticMode(unittest.TestCase):
    def test_format_diagnostic_prompt(self):
        diagnostic = DiagnosticMode()
        
        # Create mock introspection data
        introspection_data = {
            "atom_count": 1000,
            "active_goals": [
                {"name": "Goal1", "sti": 0.8},
                {"name": "Goal2", "sti": 0.7}
            ],
            "attention_summary": {
                "avg_sti": 0.3,
                "max_sti": 0.9
            }
        }
        
        prompt = diagnostic.format_diagnostic_prompt(introspection_data)
        
        # Check that the prompt contains the expected elements
        self.assertIn("AtomSpace Data", prompt)
        self.assertIn("Total atoms: 1000", prompt)
        self.assertIn("Goal1", prompt)
        self.assertIn("Goal2", prompt)
        self.assertIn("```json", prompt)
        self.assertIn("NanoCog (Diagnostic Analysis):", prompt)

if __name__ == '__main__':
    unittest.main()