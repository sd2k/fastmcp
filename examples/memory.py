"""
filesystem mcp server with basic memory capabilities.
keeps a user profile that can be updated and summarized by an llm.
"""

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from fastmcp import FastMCP

MAX_MEMORIES = 3


class Memory(BaseModel):
    """a single memory/observation about the user"""

    content: str
    timestamp: float
    importance: Annotated[int, Field(ge=1, le=5)] = Field(default=3)


class Profile(BaseModel):
    """user profile built from memories"""

    memories: list[Memory] = Field(default_factory=list, max_length=MAX_MEMORIES)
    summary: str = Field(default="")


class MemoryUpdate(BaseModel):
    """llm analysis of how to update the profile"""

    keep_indices: list[int] = Field(description="indices of memories to keep")
    new_memory: Memory = Field(description="processed version of the new memory")
    updated_summary: str = Field(description="brief summary of all memories")


memory_agent = Agent(
    "openai:gpt-4o",
    result_type=MemoryUpdate,
    system_prompt="""
    you help maintain a concise user memory profile. when given a new memory:
    1. analyze its importance relative to existing memories
    2. if we're at max capacity of memories, decide which to keep
    3. provide a brief summary of all memories
    focus on keeping the most important and relevant information.
    """,
)

mcp = FastMCP("memory", dependencies=["pydantic-ai-slim[openai]"])

PROFILE_DIR = (
    Path.home() / ".fastmcp" / os.environ.get("USER", "anon") / "memory"
).resolve()
PROFILE_DIR.mkdir(parents=True, exist_ok=True)


@mcp.tool()
async def remember(
    content: Annotated[str, Field(description="new observation/memory to store")],
    importance: Annotated[int, Field(ge=1, le=5, description="importance (1-5)")],
) -> str:
    """store a new memory/observation about the user"""
    profile_path = PROFILE_DIR / "profile.json"

    if profile_path.exists():
        profile = Profile.model_validate_json(profile_path.read_text())
    else:
        profile = Profile()

    new_memory = Memory(
        content=content,
        timestamp=datetime.now(UTC).timestamp(),
        importance=importance,
    )

    if len(profile.memories) >= MAX_MEMORIES:
        result = await memory_agent.run(
            f"""
            new memory: {content} (importance: {importance})
            
            current memories:
            {[f"{i}: {m.content} (importance: {m.importance})" 
              for i, m in enumerate(profile.memories)]}
            """
        )

        profile.memories = [profile.memories[i] for i in result.data.keep_indices]
        profile.memories.append(result.data.new_memory)
        profile.summary = result.data.updated_summary
    else:
        profile.memories.append(new_memory)

    profile_path.write_text(profile.model_dump_json(indent=2))
    return f"remembered: {content}"


@mcp.tool()
async def read_profile() -> str:
    """read and display the current memory profile"""
    profile_path = PROFILE_DIR / "profile.json"
    if not profile_path.exists():
        return "no profile found"

    profile = Profile.model_validate_json(profile_path.read_text())

    output = ["current memories:"]
    for i, memory in enumerate(profile.memories):
        output.append(
            f"{i}. {memory.content} "
            f"(importance: {memory.importance}, "
            f"timestamp: {datetime.fromtimestamp(memory.timestamp, UTC)})"
        )

    if profile.summary:
        output.append(f"\nsummary: {profile.summary}")

    return "\n".join(output)
