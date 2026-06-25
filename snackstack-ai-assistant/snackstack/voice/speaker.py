

import tempfile
import os
from snackstack.logger import get_logger
from snackstack.config import open_ai_client
import soundfile as sf
import sounddevice as sd
import uuid

logger = get_logger("speaker")


VOICE_OPTIONS = {
    "alloy":   "Neutral, professional",
    "echo":    "Male, clear and steady",
    "fable":   "British accent, expressive",
    "onyx":    "Deep male, authoritative",
    "nova":    "Female, warm and friendly",
    "shimmer": "Female, soft and gentle",
}


class VoiceSpeaker:
    """Convert text to speech and play it back."""

    def __init__(self, voice: str = "nova", speed: float = 1.0):
        self.voice = voice
        self.speed = speed
        self._out_dir = tempfile.gettempdir()

    def synthesise(self, text: str) -> str:
        """Generate an MP3 file and return its path. Each call gets a
        unique filename so previous responses are never overwritten."""
        out_path = os.path.join(self._out_dir, f"tts_{uuid.uuid4().hex[:8]}.mp3")
        try:
            resp = open_ai_client.audio.speech.create(
                model="tts-1", voice=self.voice, input=text, speed=self.speed,
            )
            resp.stream_to_file(out_path)
            logger.info("TTS saved → %s", out_path)
            return out_path
        except Exception:
            logger.exception("TTS synthesis failed")
            return ""

    def play(self, audio_file: str) -> None:
        """Play an audio file through the default output device."""
        try:
            data, sr = sf.read(audio_file)
            sd.play(data, sr)
            sd.wait()
        except Exception:
            logger.exception("Audio playback failed")

    def speak(self, text: str, play: bool = True) -> str:
        """Synthesise *text* and optionally play it. Returns the file path."""
        logger.info("Agent says: %s", text[:120])
        path = self.synthesise(text)
        if play and path:
            self.play(path)
        return path
    

# Testing
# speaker = VoiceSpeaker()

# speaker.speak("Hi How are you? ")
