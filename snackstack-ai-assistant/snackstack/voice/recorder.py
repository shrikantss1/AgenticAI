


from snackstack.logger import get_logger
from snackstack.config import open_ai_client
import sounddevice as sd
import time
import numpy as np
import io
import soundfile as sf
import tempfile
import os
import uuid


logger = get_logger("recorder")

# ═══════════════════════════════════════════════════════════
#  Speech-to-Text
# ═══════════════════════════════════════════════════════════

class VoiceRecorder:
    """Record from the microphone and transcribe with Whisper."""

    def __init__(self, sample_rate: int = 16_000):
        self.sample_rate = sample_rate

    def record(self, duration: int = 5, countdown: bool = True) -> np.ndarray:
        """Record *duration* seconds of mono audio."""
        if countdown:
            for i in range(3, 0, -1):
                logger.info("Recording starts in %d …", i)
                time.sleep(1)

        logger.info("Recording for %d s — speak now!", duration)
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        logger.info("Recording complete")
        return audio

    def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        """Send audio to Whisper and return the transcript."""
        buf = io.BytesIO()
        sf.write(buf, audio, self.sample_rate, format="WAV")
        buf.seek(0)
        buf.name = "recording.wav"

        try:
            result = open_ai_client.audio.transcriptions.create(
                model="whisper-1", file=buf, language=language,
            )
            logger.info("Transcription: %r", result.text)
            return result.text
        except Exception:
            logger.exception("Whisper transcription failed")
            return ""

    def record_and_transcribe(self, duration: int = 5) -> tuple[str, str]:
        """Full pipeline: record → save WAV → transcribe.

        Returns (wav_path, transcript).
        """
        audio = self.record(duration)
        wav_path = os.path.join(tempfile.gettempdir(), f"rec_{uuid.uuid4().hex[:8]}.wav")
        sf.write(wav_path, audio, self.sample_rate)
        text = self.transcribe(audio)
        return wav_path, text


# Testing


# recorder = VoiceRecorder()
# recorder.record_and_transcribe()
