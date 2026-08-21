# meetings_workflow

## vLLM tool calling 설정

이 앱의 의도 분류(`IntentResult`)와 회의록 수정 제안(`propose_edit`)은 `llama_index`의
`as_structured_llm(...)`으로 구조화 출력을 뽑는다. `gemma4`처럼 OpenAI의 `response_format=json_schema`
화이트리스트에 없는 모델을 쓰면 llama_index는 자동으로 **tool calling**(`tool_choice="required"`)
경로로 폴백하므로, vLLM을 아래와 같이 띄운 상태와 정확히 맞물린다.

```bash
vllm serve <model> \
  --enable-auto-tool-choice \
  --reasoning-parser gemma4 \
  --tool-call-parser gemma4
```

클라이언트(`factory.py`)에서는 `llama_index.llms.openai_like.OpenAILike`로 이 서버에 붙고,
`is_function_calling_model=True`를 명시해 tool-call 경로를 강제한다:

```python
from llama_index.llms.openai_like import OpenAILike

llm = OpenAILike(
    model="gemma4",                       # 서버의 --served-model-name과 동일해야 함
    api_base="http://localhost:8000/v1",
    api_key="EMPTY",
    is_chat_model=True,
    is_function_calling_model=True,        # 핵심: 프롬프트 JSON이 아니라 실제 tool_calls 사용
    strict=False,                          # 일부 vLLM 버전이 tool schema의 strict 필드를 거부함
)
```

이렇게 두면 `Settings.llm.as_structured_llm(SomeSchema).achat([...])` 호출 시
llama_index가 `SomeSchema`를 tool 스펙으로 변환해 `tools=[...]`, `tool_choice="required"`로 전송하고,
vLLM은 `--tool-call-parser gemma4`로 모델 출력에서 tool_calls를 파싱해 돌려준다.
결과는 `response.raw`에 이미 파싱된 pydantic 인스턴스로 들어온다(`workflow.py`의 `_intent` 참고).

임베딩(`OpenAIEmbedding`)은 gemma4가 담당하지 않으므로 별도로 `OPENAI_API_KEY`가 필요하다.

### 환경 변수

`.env.example`을 `.env`로 복사해 값을 채운다. 주요 변수:

| 변수 | 설명 |
| --- | --- |
| `VLLM_API_BASE` | vLLM OpenAI 호환 서버 주소 (`.../v1`) |
| `VLLM_API_KEY` | 임의 문자열(값 검사 안 함, 빈 문자열만 안 됨) |
| `MEETING_MODEL` | vLLM `--served-model-name`과 동일한 값 |
| `OPENAI_API_KEY` | 임베딩용 OpenAI 키 |

### 실행

`cli.py`가 상대 임포트(`from .factory import ...`)를 쓰므로, 이 저장소를
`meetings_workflow` 패키지로 상위 디렉터리에서 실행해야 한다.

```bash
pip install -r requirements.txt
cd ..   # 이 저장소(meetings_workflow/)의 부모 디렉터리로 이동
python -m meetings_workflow.cli
```
