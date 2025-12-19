import torch

class Config:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Model IDs
    MODEL_NMT = "mrheartng/idu-eng-translator"
    MODEL_TTS_IDOMA = "mrheartng/idoma-mms-tts-eng"
    MODEL_ASR_IDOMA = "mrheartng/wav2vec2-xls-r-1b-finetuned-idoma"
    MODEL_ASR_ENG = "openai/whisper-large-v3"
    MODEL_TTS_ENG = "microsoft/speecht5_tts"
    # Audio Settings
    SAMPLING_RATE = 16000
