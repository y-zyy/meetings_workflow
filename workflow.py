from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llama_index.core import Settings
from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Context, Event, StartEvent, StopEvent, Workflow, step
from pydantic import BaseModel

from .models import AssistantResult, IntentResult, PendingEdit, Relevance, Slots, TaskType
from .store import MeetingStore


INTENT_PROMPT = """당신은 회의록 시스템의 의도 분류기다. 회의 내용 QA, 요약/문서화, 액션 아이템,
후속 회의 비교, 통계/인사이트, 회의록 편집만 relevant로 분류한다. 직전 대명사나 생략을 대화로
복원해야 하면 contextual이다. 잡담과 일반 지식은 irrelevant다. task_type과 다음 슬롯을 추출한다:
date_expression(가능하면 YYYY-MM-DD, 단 지난주/이번주는 그대로), attendees, agenda, action_item,
assignee, due_expression, meeting_id. rewritten_query는 맥락 없이도 이해되게 쓴다.
현재 날짜: {today}\n대화 기록:\n{history}\n사용자 질의: {query}"""


class RoutedEvent(Event):
    query: str
    intent: IntentResult


class RagEvent(Event):
    query: str
    intent: IntentResult
    meeting_ids: list[str]


class ActionEvent(Event):
    intent: IntentResult


class EditEvent(Event):
    query: str
    intent: IntentResult
    meeting_id: str


class MeetingAssistantWorkflow(Workflow):
    def __init__(self, store: MeetingStore, llm: Any | None = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.store = store
        self.llm = llm or Settings.llm
        self.pending: dict[str, PendingEdit] = {}

    async def _intent(self, query: str, history: list[dict[str, str]]) -> IntentResult:
        history_text = "\n".join(f"{m.get('role')}: {m.get('content')}" for m in history[-12:])
        structured = self.llm.as_structured_llm(IntentResult)
        response = await structured.achat([
            ChatMessage(role="system", content="JSON 스키마에 정확히 맞춰 응답하라."),
            ChatMessage(role="user", content=INTENT_PROMPT.format(
                today=datetime.now().date().isoformat(), history=history_text, query=query
            )),
        ])
        return IntentResult.model_validate_json(response.message.content)

    @step
    async def classify_relevance(self, ctx: Context, ev: StartEvent) -> RoutedEvent | StopEvent:
        query = str(ev.get("query", "")).strip()
        history = ev.get("chat_history", [])
        if not query:
            return StopEvent(result=AssistantResult(status="need_input", answer="질문을 입력해 주세요."))
        intent = await self._intent(query, history)
        if intent.relevance == Relevance.IRRELEVANT:
            return StopEvent(result=AssistantResult(
                status="refused", answer="죄송해요, 회의록 관련 질문만 도와드릴 수 있어요."
            ))
        return RoutedEvent(query=intent.rewritten_query or query, intent=intent)

    @step
    async def route_by_task_type(self, ctx: Context, ev: RoutedEvent) -> RagEvent | ActionEvent | EditEvent | StopEvent:
        task = ev.intent.task_type
        slots = ev.intent.slots
        if task is None:
            return StopEvent(result=AssistantResult(status="need_input", answer="원하시는 회의록 작업을 조금 더 구체적으로 알려주세요."))
        matches = self.store.select(slots)
        # 액션 조회는 담당자가 핵심 식별자이며, 나머지는 회의 식별자가 필요하다.
        if task == TaskType.ACTION_ITEM_LOOKUP:
            if not slots.assignee and not slots.meeting_id and not slots.date_expression:
                return StopEvent(result=AssistantResult(
                    status="need_input", task_type=task,
                    answer="누구의 액션 아이템을 찾을까요? 담당자나 회의 일시를 알려주세요.",
                ))
            return ActionEvent(intent=ev.intent)
        if not any([slots.meeting_id, slots.date_expression, slots.attendees, slots.agenda]):
            return StopEvent(result=AssistantResult(
                status="need_input", task_type=task,
                answer="어떤 회의를 찾아드릴까요? 일시, 참석자, 회의 안건 중 하나를 알려주세요.",
            ))
        if not matches:
            return StopEvent(result=AssistantResult(status="not_found", task_type=task, answer="조건에 맞는 회의록을 찾지 못했어요."))
        ids = [r.meeting_id for r in matches if r.meeting_id]
        if task == TaskType.EDIT:
            if len(ids) != 1:
                return StopEvent(result=AssistantResult(status="need_input", task_type=task, answer="수정할 회의를 하나로 특정해 주세요."))
            return EditEvent(query=ev.query, intent=ev.intent, meeting_id=ids[0])
        return RagEvent(query=ev.query, intent=ev.intent, meeting_ids=ids)

    @step
    async def rag_search_and_answer(self, ctx: Context, ev: RagEvent) -> StopEvent:
        if self.store.index is None:
            self.store.build_index()
        retriever = self.store.index.as_retriever(similarity_top_k=max(5, len(ev.meeting_ids) * 2))
        nodes = await retriever.aretrieve(ev.query)
        nodes = [n for n in nodes if n.metadata.get("meeting_id") in ev.meeting_ids]
        context = "\n\n---\n\n".join(n.get_content() for n in nodes)
        if not context:
            # 메타데이터 필터 후 결과가 비면 선택된 원문을 직접 컨텍스트로 사용한다.
            docs = {d.doc_id: d.text for d in self.store._documents()}
            context = "\n\n---\n\n".join(docs[mid] for mid in ev.meeting_ids if mid in docs)
        prompt = (
            "다음 회의록 근거만 사용해 한국어로 답하라. 추측하지 말고, 회의별 흐름과 날짜를 명확히 하며 "
            "근거가 없으면 없다고 말하라. STATS는 계산 기준과 수치를, FOLLOWUP_COMPARE는 변화/미완료를 포함하라.\n\n"
            f"질의: {ev.query}\n\n회의록:\n{context}"
        )
        response = await self.llm.acomplete(prompt)
        return StopEvent(result=AssistantResult(
            status="ok", answer=str(response), task_type=ev.intent.task_type, sources=ev.meeting_ids
        ))

    @step
    async def action_item_lookup(self, ctx: Context, ev: ActionEvent) -> StopEvent:
        actions = self.store.find_actions(ev.intent.slots)
        if not actions:
            return StopEvent(result=AssistantResult(status="not_found", task_type=TaskType.ACTION_ITEM_LOOKUP, answer="조건에 맞는 액션 아이템이 없어요."))
        lines = [
            f"- {a.task} — 담당자: {a.assignee or '미정'}, 기한: {a.due_date or '미정'}, 상태: {a.status} ({r.occurred_at.date()}, {r.meeting_id})"
            for r, a in actions
        ]
        return StopEvent(result=AssistantResult(
            status="ok", task_type=TaskType.ACTION_ITEM_LOOKUP,
            answer="확인된 액션 아이템입니다.\n" + "\n".join(lines),
            sources=sorted({r.meeting_id for r, _ in actions if r.meeting_id}),
        ))

    @step
    async def propose_edit(self, ctx: Context, ev: EditEvent) -> StopEvent:
        record = next(r for r in self.store.records if r.meeting_id == ev.meeting_id)
        schema = type("EditProposal", (BaseModel,), {
            "__annotations__": {"field": str, "proposed": Any, "reason": str}
        })
        structured = self.llm.as_structured_llm(schema)
        response = await structured.acomplete(
            "요청에 따라 최소 범위의 수정안을 만들어라. field는 transcript, attendees, agenda, decisions 중 하나다.\n"
            f"요청: {ev.query}\n문서: {record.model_dump_json(by_alias=True, ensure_ascii=False)}"
        )
        proposal = schema.model_validate_json(response.text)
        original = getattr(record, proposal.field)
        token = uuid.uuid4().hex[:12]
        self.pending[token] = PendingEdit(
            confirmation_id=token, file_path=str(self.store.paths[ev.meeting_id]), meeting_id=ev.meeting_id,
            field=proposal.field, original=original, proposed=proposal.proposed,
            created_at=datetime.now(timezone.utc),
        )
        return StopEvent(result=AssistantResult(
            status="confirmation_required", task_type=TaskType.EDIT, confirmation_id=token,
            answer=f"수정안을 확인해 주세요.\n- 필드: {proposal.field}\n- 수정 전: {original}\n- 수정 후: {proposal.proposed}\n- 이유: {proposal.reason}\n저장하려면 confirmation_id와 함께 confirm_edit()를 호출하세요.",
            sources=[ev.meeting_id],
        ))

    def confirm_edit(self, confirmation_id: str, approved: bool) -> AssistantResult:
        pending = self.pending.pop(confirmation_id, None)
        if pending is None:
            return AssistantResult(status="not_found", answer="유효한 수정 요청을 찾지 못했어요.")
        if not approved:
            return AssistantResult(status="cancelled", task_type=TaskType.EDIT, answer="수정을 취소했어요.")
        self.store.save_field(pending.meeting_id, pending.field, pending.proposed)
        # 저장 후에는 다음 질의에서 새 인덱스를 강제 재생성해야 한다.
        self.store.index = None
        if self.store.persist_dir.exists():
            import shutil
            shutil.rmtree(self.store.persist_dir)
        return AssistantResult(status="saved", task_type=TaskType.EDIT, answer="수정 내용을 저장했어요.", sources=[pending.meeting_id])

