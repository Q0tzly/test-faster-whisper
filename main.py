import queue
import tempfile
import wave

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SECONDS = 3

audio_queue = queue.Queue()

print("Loading model...")
model = WhisperModel(
    "large-v3-turbo",
    device="cpu",
    compute_type="int8",
)
print("Model loaded!")


def audio_callback(indata, frames, time, status):
    if status:
        print(status)

    audio_queue.put(indata.copy())


buffer = []

print("Listening... (Ctrl+C to stop)")

with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype=np.int16,
    callback=audio_callback,
):

    while True:
        chunk = audio_queue.get()
        buffer.append(chunk)

        samples = sum(len(x) for x in buffer)

        if samples >= SAMPLE_RATE * CHUNK_SECONDS:
            audio = np.concatenate(buffer, axis=0)
            buffer.clear()

            volume = np.abs(audio).mean()

            if volume < 100:
                continue

            with tempfile.NamedTemporaryFile(
                suffix=".wav"
            ) as tmp:

                with wave.open(tmp.name, "wb") as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio.tobytes())

                segments, _ = model.transcribe(
                    tmp.name,
                    language="ja",
                    beam_size=1,
                    vad_filter=True,
                    condition_on_previous_text=False,
                )

                text = "".join(
                    seg.text for seg in segments
                ).strip()

                if text:
                    print(text)
