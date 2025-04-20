from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    AgentSession,
    Agent,
    RoomInputOptions,
    function_tool,
    RunContext,
)
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
import asyncio
from typing import List, Dict, Any

# Import necessary components from agents.py for Slack integration
from agents import (
    get_slack_agent_async,
    Runner,
    InMemorySessionService,
    InMemoryArtifactService,
)
from google.genai import types

# No longer using our custom voice filter
# from voice_filter import load_voice_filter

load_dotenv()

logger = logging.getLogger("myagent")
logger.setLevel(logging.INFO)


class VoiceAssistant(Agent):
    def __init__(self) -> None:
        system_prompt = """
            You are a helpful voice AI assistant.
            You are chatting with a user who is babysitting at the same time.
            Some messages will be work-related, some will be the user talking to the baby.
            Use your judgement to determine which messasages are work related.
            If the user is talking to the baby - return just the SPACE character that's it.
            Look for phrases like
            - "careful"
            - "dont touch that"
            - "stop"
            - "dont eat it"
            - "whos my little pumpkin" 
            etc
            
            IMPORTANT: When using tools, do NOT generate additional responses. The tools will handle 
            all necessary verbal responses themselves. Just call the appropriate tool and let it respond.
            """
        super().__init__(instructions=system_prompt)
        self.tasks = []

    @function_tool()
    async def check_tasks(self, context: RunContext) -> List[str]:
        """Returns a list of all current tasks.
        Use this tool when the user asks about their tasks or to-do list.
        The tool will handle all verbal responses to the user.
        """
        # Generate a spoken response about the tasks
        if len(self.tasks) == 0:
            await context.session.say("You don't have any tasks at the moment.")
        else:
            tasks_text = ", ".join(
                [f"{i+1}. {task}" for i, task in enumerate(self.tasks)]
            )
            await context.session.say(f"Here are your current tasks: {tasks_text}")

        return self.tasks

    @function_tool()
    async def add_task(self, context: RunContext, task: str) -> str:
        """
        Add a new task to the task list

        Args:
            task: The task description to add

        The tool will handle all verbal responses to the user.
        """
        self.tasks.append(task)

        # Generate a spoken response about adding the task
        await context.session.say(f"I've added '{task}' to your task list.")

        return f"Added task: {task}"

    @function_tool()
    async def remove_task(self, context: RunContext, index: int) -> str:
        """
        Remove a task by its index

        Args:
            index: The index of the task to remove (starting from 1)

        The tool will handle all verbal responses to the user.
        """
        # Convert from 1-based (user-friendly) to 0-based index
        idx = index - 1

        if 0 <= idx < len(self.tasks):
            task = self.tasks.pop(idx)

            # Generate a spoken response about removing the task
            await context.session.say(f"I've removed the task '{task}' from your list.")

            return f"Removed task: {task}"
        else:
            # Generate a spoken response for invalid index
            await context.session.say(
                f"I couldn't find task number {index}. Please give me a valid task number between 1 and {len(self.tasks)}."
            )

            return "Invalid task index"

    @function_tool()
    async def send_slack_message(
        self, context: RunContext, message: str, channel: str
    ) -> str:
        """
        Send a message to Slack

        Args:
            message: The message to send to Slack
            channel: Channel name to send the message to (must be one of: mcptest, general, random)

        The tool will handle all verbal responses to the user.
        """
        # Validate channel is in the allowed list
        allowed_channels = ["mcptest", "general", "random"]
        if channel not in allowed_channels:
            error_message = f"Invalid channel: {channel}. Must be one of: {', '.join(allowed_channels)}"
            await context.session.say(error_message)
            return error_message

        try:
            await context.session.say(
                f"I'll send that message to the {channel} Slack channel for you."
            )

            # Log the request rather than actually executing it for now
            # This can help debug if the Slack integration is causing the timeout
            logger.info(f"Slack message request: channel={channel}, message={message}")

            # The commented-out code below can be re-enabled once we confirm the basic functionality works

            # Create session services for the Slack agent
            session_service = InMemorySessionService()
            artifacts_service = InMemoryArtifactService()

            # Create a session
            session = session_service.create_session(
                state={}, app_name="slack_integration", user_id="voice_assistant_user"
            )

            # Set up the query for the Slack agent
            slack_command = f"send message '{message}' to channel {channel}"

            # Create the content object for the Slack agent
            content = types.Content(role="user", parts=[types.Part(text=slack_command)])

            # Get the Slack agent
            root_agent, exit_stack = await get_slack_agent_async()

            # Create runner for the Slack agent
            runner = Runner(
                app_name="slack_integration",
                agent=root_agent,
                artifact_service=artifacts_service,
                session_service=session_service,
            )

            # Run the Slack agent
            events_async = runner.run_async(
                session_id=session.id, user_id=session.user_id, new_message=content
            )

            # Process events
            result = "Message sent to Slack"
            async for event in events_async:
                logger.info(f"Slack event: {event}")

            # Clean up
            await exit_stack.aclose()

            await context.session.say(
                f"Message sent successfully to the {channel} channel in Slack."
            )
            return result

        except Exception as e:
            error_message = f"Error sending message to Slack: {str(e)}"
            logger.error(error_message)
            await context.session.say(
                "I encountered an error trying to send your message to Slack."
            )
            return error_message

    @function_tool()
    async def read_slack_messages(
        self, context: RunContext, channel: str, count: int = 5
    ) -> str:
        """
        Read the latest messages from a Slack channel

        Args:
            channel: Channel name to read messages from (must be one of: mcptest, general, random)
            count: Number of recent messages to retrieve (default: 5)

        The tool will handle all verbal responses to the user.
        """
        # Channel ID mapping (you need to replace these with actual channel IDs from your workspace)
        channel_id_map = {
            "mcptest": "C08NEV03L23",  # Replace with real channel ID
            "general": "CG9DQQUQ5",  # Replace with real channel ID
            "random": "CG948BT5Y",  # Replace with real channel ID
        }

        # Validate channel is in the allowed list
        allowed_channels = list(channel_id_map.keys())
        if channel not in allowed_channels:
            return f"Invalid channel: {channel}. Must be one of: {', '.join(allowed_channels)}"

        try:
            # Get the channel ID from our mapping
            channel_id = channel_id_map.get(channel)

            # Log the request
            logger.info(
                f"Slack read request: channel={channel} (ID: {channel_id}), count={count}"
            )

            # Create session services for the Slack agent
            session_service = InMemorySessionService()
            artifacts_service = InMemoryArtifactService()

            # Create a session
            session = session_service.create_session(
                state={}, app_name="slack_integration", user_id="voice_assistant_user"
            )

            # Set up the query for the Slack agent
            slack_command = (
                f"get the {count} most recent messages from channel ID {channel_id}"
            )

            # Create the content object for the Slack agent
            content = types.Content(role="user", parts=[types.Part(text=slack_command)])

            # Get the Slack agent
            root_agent, exit_stack = await get_slack_agent_async()

            # Create runner for the Slack agent
            runner = Runner(
                app_name="slack_integration",
                agent=root_agent,
                artifact_service=artifacts_service,
                session_service=session_service,
            )

            # Run the Slack agent
            events_async = runner.run_async(
                session_id=session.id, user_id=session.user_id, new_message=content
            )

            # Process events to extract and format messages
            raw_content = []
            async for event in events_async:
                logger.info(f"Slack event: {event}")
                if hasattr(event, "content") and event.content:
                    # Collect raw content from events
                    content_text = (
                        event.content.text
                        if hasattr(event.content, "text")
                        else str(event.content)
                    )
                    raw_content.append(content_text)

            # Clean up
            await exit_stack.aclose()

            # Join all raw content for processing
            combined_response = " ".join(raw_content)

            # Clean and format the messages
            clean_messages = []

            # Look for common patterns in Slack API responses
            import re

            # Extract message data using regex patterns
            # Try to find user messages in various formats
            message_patterns = [
                r'(?:user|sender):\s*(\w+),?\s*text:\s*"?([^"]+)"?',  # Matches "user: X, text: Y"
                r'(\w+):\s+"?([^"]+)"?',  # Simple "User: Message" format
                r"\d+\.\s+([^:]+):\s+(.+?)(?=\n\d+\.|\n|$)",  # Numbered list format
            ]

            for pattern in message_patterns:
                matches = re.findall(pattern, combined_response)
                if matches:
                    for match in matches:
                        if len(match) == 2:
                            username, message = match
                            # Clean up user IDs (convert U12345678 to a readable name)
                            if re.match(r"^U[A-Z0-9]{8,}$", username):
                                username = (
                                    "User"  # Replace user IDs with generic "User"
                                )
                            clean_messages.append(f"{username}: {message.strip()}")

            # If we didn't find matches with our patterns, try a simple line-by-line approach
            if not clean_messages:
                lines = combined_response.split("\n")
                for line in lines:
                    if ":" in line and len(line) > 10:
                        # Try to extract username and message
                        parts = line.split(":", 1)
                        if len(parts) == 2:
                            username, message = parts
                            # Clean up user IDs
                            if re.match(r"^U[A-Z0-9]{8,}$", username.strip()):
                                username = "User"
                            clean_messages.append(
                                f"{username.strip()}: {message.strip()}"
                            )

            # If we still have nothing, return a helpful message
            if not clean_messages:
                return f"I connected to the {channel} channel, but couldn't find any messages to read."

            # Limit to requested count and remove duplicates (keep order)
            unique_messages = []
            for msg in clean_messages:
                if msg not in unique_messages:
                    unique_messages.append(msg)

            # Take only the requested number of messages
            final_messages = unique_messages[:count]

            # Return a simple, clean string with the messages
            return f"Here are the latest messages from {channel}:\n" + "\n".join(
                final_messages
            )

        except Exception as e:
            error_message = f"Error reading messages from Slack: {str(e)}"
            logger.error(error_message)
            await context.session.say(
                "I encountered an error trying to read messages from Slack."
            )
            return error_message


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

    voice_assistant = VoiceAssistant()

    session = AgentSession(
        stt=deepgram.STT(model="nova-2-general"),
        llm=google.LLM(
            model="gemini-2.0-flash",
            # Keep on auto to allow the model to decide when to use tools
            tool_choice="auto",
        ),
        tts=elevenlabs.TTS(model="eleven_turbo_v2_5"),
        vad=silero.VAD.load(),
        turn_detection=MultilingualModel(),
    )

    logger.info("Starting agent session...")
    await session.start(
        room=ctx.room,
        agent=voice_assistant,
        room_input_options=RoomInputOptions(
            noise_cancellation=noise_cancellation_option,
        ),
    )

    await session.generate_reply(
        instructions="Greet the user and offer your assistance with a friendly tone. Mention that you can help manage tasks."
    )

    logger.info("Agent started successfully with voice filtering")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
#
