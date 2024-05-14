import json
from pathlib import Path
from typing import Optional, List
from textwrap import dedent

from phi.assistant import Assistant
from phi.tools import Toolkit
from phi.tools.calculator import Calculator
from phi.tools.duckduckgo import DuckDuckGo
from phi.tools.yfinance import YFinanceTools
from phi.tools.file import FileTools
from phi.assistant.duckdb import DuckDbAssistant
from phi.assistant.python import PythonAssistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.utils.log import logger
from phi.llm.openai import OpenAIChat
from phi.knowledge import AssistantKnowledge
from phi.embedder.openai import OpenAIEmbedder
from phi.tools.exa import ExaTools
from phi.vectordb.pgvector import PgVector2

# Set up constants
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"
cwd = Path(__file__).parent.resolve()
scratch_dir = cwd.joinpath("scratch")

# Ensure scratch directory exists
scratch_dir.mkdir(exist_ok=True, parents=True)

# Define function to create agent
def get_agent(
    llm_id: str = "gpt-4o",
    tools_config: dict = {},
    team_config: dict = {},
    user_id: Optional[str] = None,
    run_id: Optional[str] = None,
    debug_mode: bool = True,
) -> Assistant:
    logger.info(f"-*- Creating {llm_id} Agent -*-")

    # Add tools available to the Agent
    tools: List[Toolkit] = []
    extra_instructions: List[str] = []

    # Calculator tool
    calculator_config = tools_config.get("calculator", {})
    if calculator_config.get("enabled", False):
        tools.append(Calculator(**calculator_config))

    # DuckDuckGo tool
    ddg_config = tools_config.get("duckduckgo", {})
    if ddg_config.get("enabled", False):
        tools.append(DuckDuckGo(**ddg_config))

    # YFinanceTools tool
    finance_config = tools_config.get("finance", {})
    if finance_config.get("enabled", False):
        tools.append(YFinanceTools(**finance_config))

    # FileTools tool
    file_config = tools_config.get("file", {})
    if file_config.get("enabled", False):
        tools.append(FileTools(base_dir=cwd))
        extra_instructions.append(
            "You can use the `read_file` tool to read a file, `save_file` to save a file, and `list_files` to list files in the working directory."
        )

    # Add team members available to the Agent
    team: List[Assistant] = []
    for role, config in team_config.items():
        if config.get("enabled", False):
            config["base_dir"] = scratch_dir
            if role == "data_analyst":
                team.append(DuckDbAssistant(**config))
                extra_instructions.append(
                    "To answer questions about my favorite movies, delegate the task to the `Data Analyst`."
                )
            elif role == "python_assistant":
                team.append(PythonAssistant(**config))
                extra_instructions.append(
                    "To write and run python code, delegate the task to the `Python Assistant`."
                )

    # Create the Agent
    agent = Assistant(
        name="agent",
        run_id=run_id,
        user_id=user_id,
        llm=OpenAIChat(model=llm_id),
        description=dedent("""\
        You are a powerful AI Agent called `Optimus Prime v7`.
        You have access to a set of tools and a team of AI Assistants at your disposal.
        Your goal is to assist the user in the best way possible.\
        """),
        instructions=[
            "When the user sends a message, first **think** and determine if:\n"
            " - You can answer by using a tool available to you\n"
            " - You need to search the knowledge base\n"
            " - You need to search the internet\n"
            " - You need to delegate the task to a team member\n"
            " - You need to ask a clarifying question",
            "If the user asks about a topic, first ALWAYS search your knowledge base using the `search_knowledge_base` tool.",
            "If you don't find relevant information in your knowledge base, use the `duckduckgo_search` tool to search the internet.",
            "If the user asks to summarize the conversation or if you need to reference your chat history with the user, use the `get_chat_history` tool.",
            "If the user's message is unclear, ask clarifying questions to get more information.",
            "Carefully read the information you have gathered and provide a clear and concise answer to the user.",
            "Do not use phrases like 'based on my knowledge' or 'depending on the information'.",
            "You can delegate tasks to an AI Assistant in your team depending on their role and the tools available to them.",
        ],
        extra_instructions=extra_instructions,
        storage=PgAssistantStorage(table_name="agent_runs", db_url=db_url),
        knowledge_base=AssistantKnowledge(
            vector_db=PgVector2(
                db_url=db_url,
                collection="agent_documents",
                embedder=OpenAIEmbedder(model="text-embedding-3-small", dimensions=1536),
            ),
            num_documents=3,
        ),
        tools=tools,
        team=team,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
        add_chat_history_to_messages=True,
        num_history_messages=4,
        markdown=True,
        add_datetime_to_instructions=True,
        introduction=dedent("""\
        Hi, I'm Optimus Prime v7, your powerful AI Assistant. Send me on my mission boss :statue_of_liberty:\
        """),
        debug_mode=debug_mode,
    )
    return agent
