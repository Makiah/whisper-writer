from transcription import TranscriptionHandler
import whisper
from computercontroller import ComputerController

def run():
    TranscriptionHandler(\
        whisper.load_model(name="base.en", device="cuda"), \
        ComputerController()).record_and_transcribe()

if __name__ == "__main__":
    run()
