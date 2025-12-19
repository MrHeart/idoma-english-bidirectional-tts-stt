from functools import lru_cache
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, VitsModel, 
    Wav2Vec2ForCTC, Wav2Vec2Processor, WhisperProcessor, 
    WhisperForConditionalGeneration, pipeline
)
from datasets import load_dataset
import torch
from config.settings import Config

class ModelManager:
    """
    Singleton manager to load models only once.
    """
    @staticmethod
    @lru_cache(maxsize=1)
    def load_nmt_models():
        print(f"Loading NMT: {Config.MODEL_NMT}...")
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NMT)
        model = AutoModelForSeq2SeqLM.from_pretrained(Config.MODEL_NMT).to(Config.DEVICE)
        return tokenizer, model

    @staticmethod
    @lru_cache(maxsize=1)
    def load_idoma_asr():
        print(f"Loading Idoma ASR: {Config.MODEL_ASR_IDOMA}...")
        processor = Wav2Vec2Processor.from_pretrained(Config.MODEL_ASR_IDOMA)
        model = Wav2Vec2ForCTC.from_pretrained(Config.MODEL_ASR_IDOMA).to(Config.DEVICE)
        return processor, model

    @staticmethod
    @lru_cache(maxsize=1)
    def load_english_asr():
        print(f"Loading English ASR: {Config.MODEL_ASR_ENG}...")
        processor = WhisperProcessor.from_pretrained(Config.MODEL_ASR_ENG)
        model = WhisperForConditionalGeneration.from_pretrained(Config.MODEL_ASR_ENG).to(Config.DEVICE)
        return processor, model

    @staticmethod
    @lru_cache(maxsize=1)
    def load_idoma_tts():
        print(f"Loading Idoma TTS: {Config.MODEL_TTS_IDOMA}...")
        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_TTS_IDOMA)
        model = VitsModel.from_pretrained(Config.MODEL_TTS_IDOMA).to(Config.DEVICE)
        return tokenizer, model

    @staticmethod
    @lru_cache(maxsize=1)
    def load_english_tts_pipeline():
        print(f"Loading English TTS Pipeline...")
        device_id = 0 if Config.DEVICE == "cuda" else -1
        synthesiser = pipeline("text-to-speech", Config.MODEL_TTS_ENG, device=device_id)
        try:
            embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
            embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(Config.DEVICE)
        except Exception:
            embeddings = torch.randn(1, 512).to(Config.DEVICE)
        return synthesiser, embeddings
