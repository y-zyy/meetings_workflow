from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TaskType(str, Enum):
    QA = "QA"
    SUMMARY = "SUMMARY"
    ACTION_ITEM_LOOKUP = "ACTION_ITEM_LOOKUP"
    FOLLOWUP_COMPARE = "FOLLOWUP_COMPARE"
    STATS = "STATS"
    EDIT = "EDIT"


class Relevance(str, Enum):
    RELEVANT = "relevant"
    CONTEXTUAL = "contextual"
    IRRELEVANT = "irrelevant"


class ActionItem(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    task: str = Field(alias="할 일")
    assignee: str | None = Field(default=None, alias="담당자")
    due_date: date | None = Field(default=None, alias="기한")
    status: str = Field(default="미완료", alias="상태")

    @field_validator("due_date", mode="before")
    @classmethod
    def blank_due_date(cls, value: Any) -> Any:
        return None if value in (None, "", "미정") else value


class MeetingRecord(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    meeting_id: str | None = Field(default=None, alias="회의 ID")
    occurred_at: datetime = Field(alias="일시")
    attendees: list[str] = Field(alias="참석자")
    agenda: str | list[str] = Field(alias="회의 안건")
    action_items: list[ActionItem] = Field(default_factory=list, alias="Action Item")
    transcript: str = Field(default="", alias="회의 내용")
    decisions: list[str] = Field(default_factory=list, alias="결정사항")

    @field_validator("attendees", mode="before")
    @classmethod
    def split_attendees(cls, value: Any) -> Any:
        return [x.strip() for x in value.split(",")] if isinstance(value, str) else value


class Slots(BaseModel):
    date_expression: str | None = Field(default=None, description="일시 또는 상대 날짜 표현")
    attendees: list[str] = Field(default_factory=list)
    agenda: list[str] = Field(default_factory=list)
    action_item: str | None = None
    assignee: str | None = None
    due_expression: str | None = None
    meeting_id: str | None = None


class IntentResult(BaseModel):
    relevance: Relevance
    task_type: TaskType | None = None
    slots: Slots = Field(default_factory=Slots)
    rewritten_query: str = Field(description="대화 맥락을 반영해 독립적으로 이해되는 질의")


class AssistantResult(BaseModel):
    status: str
    answer: str
    task_type: TaskType | None = None
    sources: list[str] = Field(default_factory=list)
    confirmation_id: str | None = None


class PendingEdit(BaseModel):
    confirmation_id: str
    file_path: str
    meeting_id: str
    field: str
    original: Any
    proposed: Any
    created_at: datetime

