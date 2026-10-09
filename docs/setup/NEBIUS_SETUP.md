# Nebius Token Factory 설정 가이드

작성: 2026-10-05 · 소요 시간: 약 10분 · 비용: $0 (카드 등록 여부는 가입 화면에서 확인)

Tether의 에이전트 두뇌(Nemotron)는 Nebius Token Factory API로 호출한다.
해커톤 규정상 "Nebius Token Factory 또는 Nebius AI Cloud에서 동작 + NVIDIA 오픈 모델 1개 이상"이 필수이므로 이 설정 없이는 출품할 수 없다.

---

## 1. 받을 수 있는 크레딧

| 경로 | 내용 |
|---|---|
| 해커톤 전용 폼 | Token Factory **$25** |
| Builder Program 가입 | Token Factory **$25**, Tavily $25, LangSmith $100, Toloka $50, Tandem $50, Discord·오피스아워 |

- 합계 Token Factory **$50**. GPU 서버(Nebius AI Cloud) 크레딧은 없다.
- 이 프로젝트에서 쓰는 것: Token Factory(필수), LangSmith(에이전트 추적 화면, 선택).

## 2. Builder Program 가입

1. <https://dev.nebius.com/builders> 접속 → **Join** (무료)
2. 이메일로 가입하고, "무엇을 만드는지" 항목에 한 줄 적는다.
   - 예: `Tether — an agent that closes the Sim2Real gap using Nemotron and NVIDIA Newton (Nebius x NVIDIA Global AI Hackathon, Physical AI track)`
3. 이메일 인증
4. **Claim credits**에서 원하는 크레딧을 받는다. 최소한 **Token Factory**는 받고, 가능하면 **LangSmith**도 받는다.

## 3. 해커톤 전용 크레딧 추가

1. 아래 폼을 연다. 활성화 코드는 링크에 들어 있다.
   <https://nebius.com/promo-code?utm_promo_event_code=2026-devpost-global-ai-hack&utm_promo_code_type=Token_Factory&utm_promo_activation_code=NEBIUS-DEVPOST-GLOBAL26>
2. 코드를 직접 입력해야 하면 `NEBIUS-DEVPOST-GLOBAL26`을 입력한다.
3. Token Factory 콘솔의 잔액이 $50인지 확인한다.

## 4. API 키 발급

1. <https://tokenfactory.nebius.com> 로그인
2. API Keys 메뉴에서 새 키를 만든다. 이름은 예를 들어 `gapcloser-dev`로 한다.
3. **키는 채팅, 코드, git에 넣지 않는다.** 셸 환경변수로만 둔다.

```bash
echo 'export NEBIUS_API_KEY=여기에_키' >> ~/.zshrc
source ~/.zshrc
```

> 저장소의 `.env`는 `.gitignore`에 포함되어 있다. `.env`를 쓸 경우에도 커밋되지 않는다.

## 5. 동작 확인

OpenAI 호환 API라서 `openai` 클라이언트를 그대로 쓴다.

```bash
.venv/bin/pip install openai
.venv/bin/python - <<'EOF'
import os
from openai import OpenAI
client = OpenAI(base_url="https://api.tokenfactory.nebius.com/v1/", api_key=os.environ["NEBIUS_API_KEY"])
for m in client.models.list().data:
    print(m.id)
EOF
```

- 출력된 모델 ID 중 `nemotron`이 들어간 것을 확인한다. 특히 **Nano Omni**가 있는지 본다.
- `base_url`은 공식 Quickstart(<https://docs.tokenfactory.nebius.com/quickstart>)에서 확인했다. JSON 출력은 `response_format`(`json_schema`)으로 강제할 수 있다.
- 모델 목록 전체를 `docs/setup/models.txt`에 저장해 두면 다음 작업에서 바로 쓸 수 있다.

## 6. 모델 사용 계획

| 모델 | 역할 | 비용 정책 |
|---|---|---|
| Nemotron 3 Ultra 550B | 핵심 가설 판단, 최종 보고서 | 반복당 1회 이하 |
| Nemotron 3 Super 120B | 에이전트 기본 두뇌, 실험 계획, 설정 변경 | 기본값 |
| Nemotron 3 Nano 30B | 로그 요약, 반복 작업 | 자유롭게 사용 |
| Nemotron 3 Nano Omni | 실패 영상 진단 (영상 입력 지원 여부 확인 필요) | 영상은 프레임 수를 줄여 전송 |

- 단가: <https://tokenfactory.nebius.com/organization/prices> (로그인 필요, 아직 확인하지 않음)
- 개발과 테스트 중에는 mock 응답을 쓴다. `make check`는 오프라인이므로 크레딧을 쓰지 않는다.
- 잔액이 $10 아래로 내려가면 Ultra 호출을 끈다.

## 7. 저장소에서 바로 확인

```bash
.venv/bin/python -m agent.llm --provider tokenfactory --models      # 모델 목록, 역할별 선택 결과 (토큰 소모 없음)
.venv/bin/python -m agent.llm --provider tokenfactory --ping        # 아주 짧은 호출 1회
.venv/bin/python -m eval.record_demo --llm tokenfactory && make dashboard   # 데모를 Token Factory로 재녹화
```

키가 없는 동안에는 `--provider local`(Ollama의 `nemotron-3-nano`)로 같은 코드를 실험할 수 있다.

## 8. 확인 체크리스트

- [ ] Builder Program 가입과 이메일 인증
- [ ] Token Factory $25 (Builder)
- [ ] Token Factory $25 (해커톤 폼)
- [ ] API 키 발급, `NEBIUS_API_KEY` 환경변수 등록
- [ ] 모델 목록 출력 성공, Nemotron과 Nano Omni 존재 확인
- [ ] (선택) LangSmith 크레딧, `LANGSMITH_API_KEY` 등록

## 출처

- 해커톤 자료: <https://nebiusglobalaihackathon.devpost.com/resources>
- Builder Program: <https://dev.nebius.com/builders>
- Token Factory × Nemotron: <https://nebius.com/services/token-factory/nemotron>
