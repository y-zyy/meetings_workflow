from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex, load_index_from_storage

from .models import ActionItem, MeetingRecord, Slots


class MeetingStore:
    def __init__(self, data_dir: str | Path, persist_dir: str | Path | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir or self.data_dir / ".index")
        self.records: list[MeetingRecord] = []
        self.paths: dict[str, Path] = {}
        self.index: VectorStoreIndex | None = None

    def load(self) -> list[MeetingRecord]:
        self.records, self.paths = [], {}
        for path in sorted(self.data_dir.glob("*.json")):
            raw = json.loads(path.read_text(encoding="utf-8"))
            items = raw if isinstance(raw, list) else [raw]
            for offset, item in enumerate(items):
                record = MeetingRecord.model_validate(item)
                record.meeting_id = record.meeting_id or f"{path.stem}-{offset + 1}"
                self.records.append(record)
                self.paths[record.meeting_id] = path
        return self.records

    def _documents(self) -> list[Document]:
        documents = []
        for record in self.records:
            agenda = ", ".join(record.agenda) if isinstance(record.agenda, list) else record.agenda
            action_text = "\n".join(
                f"- {a.task} / 담당자: {a.assignee or '미정'} / 기한: {a.due_date or '미정'} / 상태: {a.status}"
                for a in record.action_items
            )
            text = (
                f"일시: {record.occurred_at.isoformat()}\n참석자: {', '.join(record.attendees)}\n"
                f"회의 안건: {agenda}\n회의 내용:\n{record.transcript}\n"
                f"결정사항:\n" + "\n".join(record.decisions) + f"\nAction Item:\n{action_text}"
            )
            documents.append(Document(
                text=text,
                doc_id=record.meeting_id,
                metadata={
                    "meeting_id": record.meeting_id,
                    "occurred_at": record.occurred_at.isoformat(),
                    "attendees": "|".join(record.attendees),
                    "agenda": agenda,
                    "source_file": self.paths[record.meeting_id].name,
                },
            ))
        return documents

    def build_index(self, force: bool = False) -> VectorStoreIndex:
        if not self.records:
            self.load()
        if self.persist_dir.exists() and not force:
            storage = StorageContext.from_defaults(persist_dir=str(self.persist_dir))
            self.index = load_index_from_storage(storage)
        else:
            self.index = VectorStoreIndex.from_documents(self._documents(), show_progress=False)
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self.index.storage_context.persist(persist_dir=str(self.persist_dir))
        return self.index

    def select(self, slots: Slots) -> list[MeetingRecord]:
        selected = self.records
        if slots.meeting_id:
            selected = [r for r in selected if r.meeting_id == slots.meeting_id]
        if slots.attendees:
            selected = [r for r in selected if any(a in r.attendees for a in slots.attendees)]
        if slots.agenda:
            selected = [r for r in selected if any(k.lower() in str(r.agenda).lower() for k in slots.agenda)]
        if slots.date_expression:
            selected = self._date_filter(selected, slots.date_expression)
        return selected

    @staticmethod
    def _date_filter(records: Iterable[MeetingRecord], expression: str) -> list[MeetingRecord]:
        today = date.today()
        if expression == "지난주":
            start = today - timedelta(days=today.weekday() + 7)
            end = start + timedelta(days=6)
        elif expression in {"이번주", "이번 주"}:
            start = today - timedelta(days=today.weekday())
            end = start + timedelta(days=6)
        else:
            try:
                target = date.fromisoformat(expression[:10])
                start = end = target
            except ValueError:
                return list(records)
        return [r for r in records if start <= r.occurred_at.date() <= end]

    def find_actions(self, slots: Slots) -> list[tuple[MeetingRecord, ActionItem]]:
        results = []
        for record in self.select(slots):
            for action in record.action_items:
                if slots.assignee and action.assignee != slots.assignee:
                    continue
                if slots.action_item and slots.action_item.lower() not in action.task.lower():
                    continue
                if slots.due_expression in {"이번주", "이번 주"} and action.due_date:
                    today = date.today()
                    if not (today <= action.due_date <= today + timedelta(days=7)):
                        continue
                results.append((record, action))
        return results

    def save_field(self, meeting_id: str, field: str, value: object) -> None:
        path = self.paths[meeting_id]
        raw = json.loads(path.read_text(encoding="utf-8"))
        key_map = {"transcript": "회의 내용", "attendees": "참석자", "agenda": "회의 안건", "decisions": "결정사항"}
        json_key = key_map.get(field, field)
        if isinstance(raw, list):
            for item in raw:
                if item.get("회의 ID") == meeting_id:
                    item[json_key] = value
                    break
        else:
            raw[json_key] = value
        path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
        self.load()

