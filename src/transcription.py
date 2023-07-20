import traceback
import numpy as np
import sounddevice as sd
import webrtcvad
import traceback
from computercontroller import ComputerController

class TranscriptionHandler:
    def __init__(self, model, controller: ComputerController):
        self.model = model
        self.controller = controller

    def process_transcription(self, transcription):
        if transcription.endswith('.'):
            transcription = transcription[:-1]
        transcription = transcription.lower()
        self.controller.process(transcription)

    def transcribe_recording(self, recording):
        # https://github.com/openai/whisper/discussions/908
        audio_data = np.array(recording, dtype=np.int16).flatten().astype(np.float32) / 32768.0 
        print('Transcribing audio...')
        response = self.model.transcribe(audio=audio_data,
                                    language=None,
                                    verbose=True,
                                    initial_prompt=None,
                                    condition_on_previous_text=False,
                                    temperature=0.0,)
        result = response.get('text')
        print('Transcription:', result)
        
        self.process_transcription(result.strip()) if result else ''

    """
    Record audio from the microphone and transcribe it using the OpenAI API.
    Recording stops when the user stops speaking.
    """
    def record_and_transcribe(self):
        sample_rate = 16000
        frame_duration = 30  # 30ms, supported values: 10, 20, 30
        done_speaking_silence_duration = 300  # ms
        acceptable_mid_speaking_pause_duration = 150
        minimum_speaking_duration = 400

        def to_frames(duration):
            return duration // frame_duration

        vad = webrtcvad.Vad(3)  # Aggressiveness mode: 3 (highest)
        buffer = []
        recording = []
        num_speech_not_detected_frames = 0
        speaking_frames = 0
        done_speaking_silence_frames = to_frames(done_speaking_silence_duration)
        acceptable_mid_speaking_pause_frames = to_frames(acceptable_mid_speaking_pause_duration)
        minimum_speaking_frames = to_frames(minimum_speaking_duration)

        try:
            print('Recording...')
            with sd.InputStream(samplerate=sample_rate, channels=1, dtype='int16', blocksize=sample_rate * frame_duration // 1000,
                                callback=lambda indata, frames, time, status: buffer.extend(indata[:, 0])): # https://blog.furas.pl/python-what-means-0-and-1-in-numpy-or-pandas-gb.html
                while True:
                    # print("Running")
                    if len(buffer) < sample_rate * frame_duration // 1000:
                        continue

                    # buffer only contains unprocessed frames
                    frame = buffer[:sample_rate * frame_duration // 1000]
                    buffer = buffer[sample_rate * frame_duration // 1000:]

                    if vad.is_speech(np.array(frame).tobytes(), sample_rate):
                        print("Speech detected!")
                        num_speech_not_detected_frames = 0
                        speaking_frames += 1
                        recording.extend(frame)
                    else:
                        if len(recording) == 0:
                            continue

                        num_speech_not_detected_frames += 1

                        if num_speech_not_detected_frames < acceptable_mid_speaking_pause_frames:
                            print(f"Paused: frames = {num_speech_not_detected_frames} will stop at {acceptable_mid_speaking_pause_frames}")
                            continue

                        if speaking_frames < minimum_speaking_frames:
                            print(f"Not enough speaking time (got {speaking_frames}, needed {minimum_speaking_frames}), ignoring that last bit")
                            recording = []
                            speaking_frames = 0
                        elif num_speech_not_detected_frames >= done_speaking_silence_frames:
                            print(f"Long enough, length is {speaking_frames} required is {minimum_speaking_frames}")
                            self.transcribe_recording(recording)
                            recording = []
                            speaking_frames = 0

        except Exception as e:
            traceback.print_exc()

