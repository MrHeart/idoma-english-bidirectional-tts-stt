import gradio as gr
from src.services.asr_service import ASRService
from src.services.nmt_service import TranslationService
from src.services.tts_service import TTSService

# Initialize Services
asr = ASRService()
nmt = TranslationService()
tts = TTSService()

def process_pipeline(audio_path, direction):
    """Orchestrates ASR -> NMT -> TTS"""
    try:
        source_lang = "English" if direction == "English to Idoma" else "Idoma"
        target_lang = "Idoma" if direction == "English to Idoma" else "English"
        # 1. ASR
        transcribed_text = asr.transcribe(audio_path, source_lang)
        # 2. Translation
        translated_text = nmt.translate(transcribed_text, source_lang)
        # 3. TTS
        audio_out = tts.synthesize(translated_text, target_lang)
        return transcribed_text, translated_text, audio_out
    except Exception as e:
        return f"Error: {e}", "", None

# Gradio UI (to be expanded as needed)
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Idoma Translator App")
        direction = gr.Radio(["English to Idoma", "Idoma to English"], label="Translation Direction")
        audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Input Audio")
        transcribed = gr.Textbox(label="Transcribed Text")
        translated = gr.Textbox(label="Translated Text")
        audio_output = gr.Audio(label="Synthesized Audio")
        btn = gr.Button("Translate")
        btn.click(process_pipeline, inputs=[audio_input, direction], outputs=[transcribed, translated, audio_output])
    demo.launch()

if __name__ == "__main__":
    main()
