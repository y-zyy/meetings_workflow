from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai_like import OpenAILike

from .store import MeetingStore
from .workflow import MeetingAssistantWorkflow


def _build_llm() -> OpenAILike:
    """vLLM이 서빙하는 OpenAI 호환 엔드포인트에 tool calling 모드로 연결한다.

    서버가 --enable-auto-tool-choice --tool-call-parser gemma4
    --reasoning-parser gemma4 로 떠 있다는 전제 하에, is_function_calling_model=True
    를 명시해야 llama_index가 (프롬프트로 JSON을 흉내내는 대신) 실제
    OpenAI 스타일 tool_calls 를 사용해 구조화 출력을 만들어낸다. is_json_schema_supported
    체크에 걸리지 않는 커스텀 모델명이라 기본적으로도 function-calling 경로로 폴백하지만,
    명시적으로 켜 두는 편이 서버 설정과 의도를 맞추는 데 안전하다.
    """
    return OpenAILike(
        model=os.getenv("MEETING_MODEL", "gemma4"),
        api_base=os.getenv("VLLM_API_BASE", "http://localhost:8000/v1"),
        api_key=os.getenv("VLLM_API_KEY", "EMPTY"),
        temperature=0,
        context_window=int(os.getenv("MEETING_CONTEXT_WINDOW", "8192")),
        is_chat_model=True,
        is_function_calling_model=True,
        # vLLM 구버전 중 tool schema의 strict 필드를 거부하는 경우가 있어 방어적으로 끈다.
        strict=False,
        timeout=float(os.getenv("MEETING_LLM_TIMEOUT", "120")),
    )


def create_workflow(data_dir: str | Path | None = None) -> MeetingAssistantWorkflow:
    load_dotenv()
    Settings.llm = _build_llm()
    # 임베딩은 별도 설정이 없는 한 여전히 OpenAI 공식 API(OPENAI_API_KEY)를 사용한다.
    # vLLM으로 서빙 중인 gemma4는 임베딩 모델이 아니므로 대상에서 제외했다.
    Settings.embed_model = OpenAIEmbedding(model=os.getenv("MEETING_EMBED_MODEL", "text-embedding-3-small"))
    store = MeetingStore(data_dir or os.getenv("MEETING_DATA_DIR", "./data"))
    store.load()
    return MeetingAssistantWorkflow(store=store, llm=Settings.llm, timeout=120, verbose=False)
