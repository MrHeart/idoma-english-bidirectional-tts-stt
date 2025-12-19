import unittest
from unittest.mock import patch, MagicMock
from src.services.nmt_service import TranslationService

class TestTranslationService(unittest.TestCase):
    @patch('src.services.nmt_service.ModelManager')
    def test_translation_flow(self, mock_manager):
        mock_tokenizer = MagicMock()
        mock_model = MagicMock()
        mock_manager.load_nmt_models.return_value = (mock_tokenizer, mock_model)
        mock_model.generate.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = "Ole" # 'Welcome' in Idoma
        service = TranslationService()
        result = service.translate("Welcome", "English")
        self.assertEqual(result, "Ole")
        mock_tokenizer.decode.assert_called()

if __name__ == '__main__':
    unittest.main()
