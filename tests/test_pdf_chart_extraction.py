import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock imports before loading the module
sys.modules['pdf2image'] = MagicMock()
sys.modules['pdfplumber'] = MagicMock()
sys.modules['pytesseract'] = MagicMock()

from ingestion.chunking import extract_chart_descriptions_from_pdf

class TestPDFChartExtraction(unittest.TestCase):
    
    @patch('ingestion.chunking.generate_image_description')
    @patch('ingestion.chunking.convert_from_path')
    @patch('ingestion.chunking.pdfplumber')
    def test_extract_chart_descriptions(self, mock_pdfplumber, mock_convert, mock_gen_desc):
        # Mock pdfplumber to return a page with an image
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        mock_page.images = [{'width': 200, 'height': 200}] # Large enough image
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # Mock convert_from_path (pdf2image)
        mock_image = MagicMock()
        mock_convert.return_value = [mock_image]
        
        # Mock generate_image_description
        mock_gen_desc.return_value = "Chart description"
        
        # Run extraction
        descriptions = extract_chart_descriptions_from_pdf("dummy.pdf")
        
        # Verify
        self.assertEqual(len(descriptions), 1)
        self.assertIn(1, descriptions)
        self.assertIn("Chart description", descriptions[1])
        
        # Verify it skipped if image is small
        mock_page.images = [{'width': 50, 'height': 50}]
        descriptions = extract_chart_descriptions_from_pdf("dummy.pdf")
        self.assertEqual(len(descriptions), 0)

if __name__ == '__main__':
    unittest.main()
