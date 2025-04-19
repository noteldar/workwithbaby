from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentSession, Agent, RoomInputOptions
from livekit.plugins import (
    deepgram,
    noise_cancellation,
    silero,
    google,
    elevenlabs,
)
from livekit.plugins.turn_detector.multilingual import MultilingualModel
import logging
import os

# No longer using our custom voice filter
# from voice_filter import load_voice_filter

load_dotenv()

logger = logging.getLogger("myagent")
logger.setLevel(logging.INFO)


class VoiceAssistant(Agent):
    def __init__(self) -> None:
        system_prompt = """You are a helpful voice AI assistant."""
        super().__init__(instructions=system_prompt)


async def entrypoint(ctx: agents.JobContext):
    logger.info("starting entrypoint")

    await ctx.connect()

    # Check if voice sample exists (for info only)
    voice_sample_path = os.path.abspath("voice.wav")
    if os.path.exists(voice_sample_path):
        logger.info(f"Voice sample found at: {voice_sample_path}")
    else:
        logger.warning(f"Voice sample not found at: {voice_sample_path}")

    # Use the built-in Background Voice Cancellation (BVC) from LiveKit
    # This will help filter out background voices (like baby voices)
    # while preserving the main speaker's voice
    logger.info("Using LiveKit's built-in BVC (Background Voice Cancellation)")
    noise_cancellation_option = noise_cancellation.BVC()

    session = AgentSession(
        stt=deepgram.STT(model="nova-2-general"),
        llm=google.LLM(model="gemini-2.0-flash", tool_choice="required"),
        tts=elevenlabs.TTS(model="eleven_turbo_v2_5"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    logger.info("Starting agent session...")
    await session.start(
        room=ctx.room,
        agent=VoiceAssistant(),
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation_option,
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance with a friendly tone."
    )

    logger.info("Agent started successfully with voice filtering")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
#
