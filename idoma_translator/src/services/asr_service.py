import torch
import librosa
from src.model_loader import ModelManager
from config.settings import Config

class ASRService:
    def transcribe(self, audio_path: str, language: str) -> str:
        """Dispatches transcription to the correct model based on language."""
        if not audio_path:
            return ""
        audio, _ = librosa.load(audio_path, sr=Config.SAMPLING_RATE)
        if language == "Idoma":
            return self._transcribe_idoma(audio)
        else:
            return self._transcribe_english(audio)

    def _transcribe_idoma(self, audio):
        processor, model = ModelManager.load_idoma_asr()
        inputs = processor(audio, sampling_rate=Config.SAMPLING_RATE, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(inputs.input_values.to(Config.DEVICE)).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0]

    def _transcribe_english(self, audio):
        processor, model = ModelManager.load_english_asr()
        features = processor(audio, sampling_rate=Config.SAMPLING_RATE, return_tensors="pt").input_features.to(Config.DEVICE)
        with torch.no_grad():
            predicted_ids = model.generate(features)
        return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
