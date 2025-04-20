# ./adk_agent_samples/mcp_agent/agent.py
import asyncio
import os
from dotenv import load_dotenv
from google.genai import types
from google.adk.agents.llm_agent import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts.in_memory_artifact_service import (
    InMemoryArtifactService,
)  # Optional
from google.adk.tools.mcp_tool.mcp_toolset import (
    MCPToolset,
    SseServerParams,
    StdioServerParameters,
)

# Load environment variables from .env file in the parent directory
# Place this near the top, before using env vars like API keys
load_dotenv()


# --- Step 1: Import Tools from MCP Server ---
async def get_filesystem_tools_async():
    """Gets tools from the File System MCP Server."""
    print("Attempting to connect to MCP Filesystem server...")
    tools, exit_stack = await MCPToolset.from_server(
        # Use StdioServerParameters for local process communication
        connection_params=StdioServerParameters(
            command="npx",  # Command to run the server
            args=[
                "-y",  # Arguments for the command
                "@modelcontextprotocol/server-filesystem",
                # TODO: IMPORTANT! Change the path below to an ABSOLUTE path on your system.
                "/home/eldar/workwithbaby",
            ],
        )
        # For remote servers, you would use SseServerParams instead:
        # connection_params=SseServerParams(url="http://remote-server:port/path", headers={...})
    )
    print("MCP Filesystem Toolset created successfully.")
    # MCP requires maintaining a connection to the local MCP Server.
    # exit_stack manages the cleanup of this connection.
    return tools, exit_stack


async def get_slack_tools_async():
    """Gets tools from the Slack MCP Server."""
    print("Attempting to connect to MCP Slack server...")

    # Environment variables needed for Slack MCP server
    # Make sure these are set in your .env file:
    # - SLACK_BOT_TOKEN: The Bot User OAuth Token starting with `xoxb-`
    #   Required scopes for reading messages:
    #   - channels:history - Read messages in public channels
    #   - groups:history - Read messages in private channels
    #   - im:history - Read direct messages
    #   - mpim:history - Read messages in group direct messages
    #   - channels:read - View basic info about public channels
    #   - groups:read - View basic info about private channels
    # - SLACK_TEAM_ID: Your Slack workspace ID starting with `T`
    # - SLACK_CHANNEL_IDS: Optional comma-separated list of channel IDs

    tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-slack"],
            env={
                "SLACK_BOT_TOKEN": os.getenv("SLACK_BOT_TOKEN"),
                "SLACK_TEAM_ID": os.getenv("SLACK_TEAM_ID"),
                "SLACK_CHANNEL_IDS": os.getenv("SLACK_CHANNEL_IDS", ""),
            },
        )
    )
    print("MCP Slack Toolset created successfully.")
    return tools, exit_stack


# --- Step 2: Agent Definition ---
async def get_filesystem_agent_async():
    """Creates an ADK Agent equipped with tools from the Filesystem MCP Server."""
    tools, exit_stack = await get_filesystem_tools_async()
    print(f"Fetched {len(tools)} tools from Filesystem MCP server.")
    root_agent = LlmAgent(
        model="gemini-2.0-flash",  # Adjust model name if needed based on availability
        name="filesystem_assistant",
        instruction="Help user interact with the local filesystem using available tools.",
        tools=tools,  # Provide the MCP tools to the ADK agent
    )
    return root_agent, exit_stack


async def get_slack_agent_async():
    """Creates an ADK Agent equipped with tools from the Slack MCP Server."""
    tools, exit_stack = await get_slack_tools_async()
    print(f"Fetched {len(tools)} tools from Slack MCP server.")
    root_agent = LlmAgent(
        model="gemini-2.0-flash",  # Adjust model name if needed based on availability
        name="slack_assistant",
        instruction="Help user interact with the Slack workspace using available tools.",
        tools=tools,  # Provide the MCP tools to the ADK agent
    )
    return root_agent, exit_stack


async def get_combined_agent_async():
    """Creates an ADK Agent equipped with both Filesystem and Slack MCP tools."""
    print("Setting up combined agent with both Filesystem and Slack tools...")

    # Get both sets of tools
    filesystem_tools, filesystem_exit_stack = await get_filesystem_tools_async()
    slack_tools, slack_exit_stack = await get_slack_tools_async()

    # Combine the tools from both sources
    combined_tools = filesystem_tools + slack_tools
    print(
        f"Combined {len(filesystem_tools)} filesystem tools and {len(slack_tools)} Slack tools."
    )

    # Create an exit stack that will close both connections when done
    from contextlib import AsyncExitStack

    combined_exit_stack = AsyncExitStack()
    await combined_exit_stack.enter_async_context(filesystem_exit_stack)
    await combined_exit_stack.enter_async_context(slack_exit_stack)

    # Create agent with combined tools
    root_agent = LlmAgent(
        model="gemini-2.0-flash",
        name="combined_assistant",
        instruction="Help user interact with both the local filesystem and Slack workspace using available tools.",
        tools=combined_tools,
    )

    return root_agent, combined_exit_stack


# --- Step 3: Main Execution Logic ---
async def filesystem_main():
    session_service = InMemorySessionService()
    # Artifact service might not be needed for this example
    artifacts_service = InMemoryArtifactService()

    session = session_service.create_session(
        state={}, app_name="mcp_filesystem_app", user_id="user_fs"
    )

    # TODO: Change the query to be relevant to YOUR specified folder.
    # e.g., "list files in the 'documents' subfolder" or "read the file 'notes.txt'"
    query = "list files and folders in the tasks/ folder"
    print(f"User Query: '{query}'")
    content = types.Content(role="user", parts=[types.Part(text=query)])

    root_agent, exit_stack = await get_filesystem_agent_async()

    runner = Runner(
        app_name="mcp_filesystem_app",
        agent=root_agent,
        artifact_service=artifacts_service,  # Optional
        session_service=session_service,
    )

    print("Running agent...")
    events_async = runner.run_async(
        session_id=session.id, user_id=session.user_id, new_message=content
    )

    async for event in events_async:
        print(f"Event received: {event}")

    # Crucial Cleanup: Ensure the MCP server process connection is closed.
    print("Closing MCP server connection...")
    await exit_stack.aclose()
    print("Cleanup complete.")


async def slack_main():
    session_service = InMemorySessionService()
    artifacts_service = InMemoryArtifactService()

    session = session_service.create_session(
        state={}, app_name="mcp_slack_app", user_id="user_slack"
    )

    # Example query for Slack
    query = "list the channels in my Slack workspace"
    print(f"User Query: '{query}'")
    content = types.Content(role="user", parts=[types.Part(text=query)])

    root_agent, exit_stack = await get_slack_agent_async()

    runner = Runner(
        app_name="mcp_slack_app",
        agent=root_agent,
        artifact_service=artifacts_service,
        session_service=session_service,
    )

    print("Running Slack agent...")
    events_async = runner.run_async(
        session_id=session.id, user_id=session.user_id, new_message=content
    )

    async for event in events_async:
        print(f"Event received: {event}")

    print("Closing Slack MCP server connection...")
    await exit_stack.aclose()
    print("Cleanup complete.")


async def slack_read_messages():
    """Example function to specifically read messages from a Slack channel."""
    session_service = InMemorySessionService()
    artifacts_service = InMemoryArtifactService()

    session = session_service.create_session(
        state={}, app_name="mcp_slack_read", user_id="user_slack_read"
    )

    # First, get a list of all channels to find the ID
    print("Getting channel list first...")
    list_query = "list all slack channels and show their IDs"
    list_content = types.Content(role="user", parts=[types.Part(text=list_query)])

    root_agent, exit_stack = await get_slack_agent_async()

    runner = Runner(
        app_name="mcp_slack_read",
        agent=root_agent,
        artifact_service=artifacts_service,
        session_service=session_service,
    )

    print("Listing channels to find channel ID...")
    list_events = runner.run_async(
        session_id=session.id, user_id=session.user_id, new_message=list_content
    )

    # Print channel list to help user find channel IDs
    async for event in list_events:
        print(f"Channel list info: {event}")

    # Now read messages using channel ID instead of name
    channel_id = "C08NEV03L23"

    # Query using ID instead of name
    query = f"get the 5 most recent messages from channel ID {channel_id}"
    print(f"User Query: '{query}'")
    content = types.Content(role="user", parts=[types.Part(text=query)])

    # Create new session for message fetching
    session = session_service.create_session(
        state={}, app_name="mcp_slack_read", user_id="user_slack_read"
    )

    # Re-use the existing runner with a new session
    print(f"Reading messages from channel ID {channel_id}...")
    events_async = runner.run_async(
        session_id=session.id, user_id=session.user_id, new_message=content
    )

    print("\nMessage content:")
    async for event in events_async:
        print(f"Message data: {event}")

    print("Closing Slack MCP server connection...")
    await exit_stack.aclose()
    print("Cleanup complete.")


async def combined_main():
    """Run an agent with access to both Filesystem and Slack tools."""
    session_service = InMemorySessionService()
    artifacts_service = InMemoryArtifactService()

    session = session_service.create_session(
        state={}, app_name="mcp_combined_app", user_id="user_combined"
    )

    # Example query that could use both sets of tools
    query = "find text files in the tasks/ folder and send me attachment in slack"
    print(f"User Query: '{query}'")
    content = types.Content(role="user", parts=[types.Part(text=query)])

    root_agent, exit_stack = await get_combined_agent_async()

    runner = Runner(
        app_name="mcp_combined_app",
        agent=root_agent,
        artifact_service=artifacts_service,
        session_service=session_service,
    )

    print("Running combined agent...")
    events_async = runner.run_async(
        session_id=session.id, user_id=session.user_id, new_message=content
    )

    async for event in events_async:
        print(f"Event received: {event}")

    print("Closing MCP server connections...")
    await exit_stack.aclose()
    print("Cleanup complete.")


if __name__ == "__main__":
    try:
        # Choose which agent to run by uncommenting one of these lines:
        # asyncio.run(filesystem_main())
        # asyncio.run(slack_main())
        # asyncio.run(combined_main())

        # For testing Slack message reading specifically:
        asyncio.run(slack_read_messages())
    except Exception as e:
        print(f"An error occurred: {e}")
