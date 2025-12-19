import torch
import numpy as np
import scipy.io.wavfile as wavfile
import tempfile
from src.model_loader import ModelManager
from config.settings import Config

class TTSService:
    def synthesize(self, text: str, target_lang: str) -> str:
        if not text:
            return None
        if target_lang == "Idoma":
            return self._generate_idoma(text)
        else:
            return self._generate_english(text)

    def _generate_idoma(self, text):
        tokenizer, model = ModelManager.load_idoma_tts()
        inputs = tokenizer(text=text, return_tensors="pt")
        input_ids = inputs["input_ids"].to(Config.DEVICE)
        with torch.no_grad():
            outputs = model(input_ids)
        waveform = outputs.waveform[0].cpu().numpy()
        return self._save_audio(waveform, Config.SAMPLING_RATE)

    def _generate_english(self, text):
        pipeline, embeddings = ModelManager.load_english_tts_pipeline()
        speech = pipeline(text, forward_params={"speaker_embeddings": embeddings})
        return self._save_audio(speech["audio"], speech["sampling_rate"])

    def _save_audio(self, audio_data: np.ndarray, rate: int) -> str:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        wavfile.write(temp_file.name, rate, audio_data)
        return temp_file.name
