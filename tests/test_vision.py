import os
import unittest
from unittest.mock import MagicMock, patch
import base64

from rag.llm import generate_image_description
from config.settings import get_settings

class TestVision(unittest.TestCase):
    
    @patch('rag.llm.ChatCompletionsClient')
    @patch('rag.llm.get_settings')
    def test_generate_image_description(self, mock_get_settings, mock_client_cls):
        # Mock settings
        settings = MagicMock()
        settings.enable_chart_description = True
        settings.github_vision_model = "gpt-4o"
        settings.github_token = "fake_token"
        settings.azure_openai_endpoint = None
        mock_get_settings.return_value = settings
        
        # Mock client
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        
        # Mock response
        mock_response = MagicMock()
        mock_choice = MagicMock()
        mock_message = MagicMock()
        mock_message.content = "This is a chart showing sales data."
        mock_choice.message = mock_message
        mock_response.choices = [mock_choice]
        mock_client.complete.return_value = mock_response
        
        # Call function
        image_data = b"fake_image_data"
        desc = generate_image_description(image_data)
        
        # Verify
        self.assertEqual(desc, "This is a chart showing sales data.")
        
        # Verify call args
        mock_client.complete.assert_called_once()
        call_args = mock_client.complete.call_args
        self.assertEqual(call_args.kwargs['model'], "gpt-4o")
        messages = call_args.kwargs['messages']
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[1]['role'], 'user')
        content = messages[1]['content']
        self.assertIsInstance(content, list)
        self.assertEqual(content[0]['type'], 'text')
        self.assertEqual(content[1]['type'], 'image_url')

    @patch('rag.llm.get_settings')
    def test_generate_image_description_disabled(self, mock_get_settings):
        # Mock settings
        settings = MagicMock()
        settings.enable_chart_description = False
        mock_get_settings.return_value = settings
        
        # Call function
        desc = generate_image_description(b"data")
        self.assertEqual(desc, "")

if __name__ == '__main__':
    unittest.main()
