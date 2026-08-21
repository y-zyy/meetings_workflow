from __future__ import annotations

import asyncio
import json

from .factory import create_workflow


async def chat() -> None:
    workflow = create_workflow()
    history: list[dict[str, str]] = []
    print("회의록 어시스턴트입니다. 종료하려면 /quit 를 입력하세요.")
    while True:
        query = input("사용자> ").strip()
        if query == "/quit":
            return
        result = await workflow.run(query=query, chat_history=history)
        payload = result.model_dump(mode="json")
        print("어시스턴트>", payload["answer"])
        if payload.get("confirmation_id"):
            answer = input("저장할까요? (y/N)> ").lower() == "y"
            saved = workflow.confirm_edit(payload["confirmation_id"], answer)
            print("어시스턴트>", saved.answer)
        history.extend([{"role": "user", "content": query}, {"role": "assistant", "content": payload["answer"]}])


def main() -> None:
    asyncio.run(chat())


if __name__ == "__main__":
    main()

