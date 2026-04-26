"""
TTS Synthesizer
Converts a text script to an MP3 audio file using gTTS (free).

gTTS = Google Text-to-Speech via the unofficial Google Translate TTS API.
No API key needed. Quality is acceptable for development/demo.

For production: swap in ElevenLabs API (set ELEVENLABS_API_KEY in .env).
The interface is the same — just swap the implementation in synthesize().
"""
import os
import hashlib
import structlog
from pathlib import Path

log = structlog.get_logger()

AUDIO_DIR = Path("./audio_cache")
AUDIO_DIR.mkdir(exist_ok=True)


def _audio_path(script: str) -> Path:
    """Generate a deterministic filename from the script content."""
    hash_str = hashlib.md5(script.encode()).hexdigest()[:12]
    return AUDIO_DIR / f"narration_{hash_str}.mp3"


async def synthesize(script: str, lang: str = "en") -> str | None:
    """
    Convert script text to MP3 audio.

    Returns the file path on success, None on failure.
    gTTS is synchronous, so we run it in a thread to avoid blocking.
    """
    audio_path = _audio_path(script)

    # Return cached file if already generated
    if audio_path.exists():
        log.debug("tts_cache_hit", path=str(audio_path))
        return str(audio_path)

    try:
        # Run gTTS in a thread pool to avoid blocking the async event loop
        import asyncio
        from functools import partial

        def _run_gtts():
            from gtts import gTTS
            tts = gTTS(text=script, lang=lang, slow=False)
            tts.save(str(audio_path))

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _run_gtts)

        log.info("tts_generated", path=str(audio_path), chars=len(script))
        return str(audio_path)

    except Exception as e:
        log.warning("tts_failed", error=str(e))
        return None
