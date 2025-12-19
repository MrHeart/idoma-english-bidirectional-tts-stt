import torch
from src.model_loader import ModelManager
from config.settings import Config

class TranslationService:
    def translate(self, text: str, source_lang: str) -> str:
        if not text:
            return ""
        tokenizer, model = ModelManager.load_nmt_models()
        if source_lang == "Idoma":
            src_code, tgt_code = "idu_Latn", "eng_Latn"
        else:
            src_code, tgt_code = "eng_Latn", "idu_Latn"
        tokenizer.src_lang = src_code
        inputs = tokenizer(text, return_tensors="pt").to(Config.DEVICE)
        tgt_token_id = tokenizer.convert_tokens_to_ids(tgt_code)
        outputs = model.generate(**inputs, forced_bos_token_id=tgt_token_id, max_length=256)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
