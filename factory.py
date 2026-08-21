from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from .store import MeetingStore
from .workflow import MeetingAssistantWorkflow


def create_workflow(data_dir: str | Path | None = None) -> MeetingAssistantWorkflow:
    load_dotenv()
    Settings.llm = OpenAI(model=os.getenv("MEETING_MODEL", "gpt-4.1-mini"), temperature=0)
    Settings.embed_model = OpenAIEmbedding(model=os.getenv("MEETING_EMBED_MODEL", "text-embedding-3-small"))
    store = MeetingStore(data_dir or os.getenv("MEETING_DATA_DIR", "./data"))
    store.load()
    return MeetingAssistantWorkflow(store=store, llm=Settings.llm, timeout=120, verbose=False)

