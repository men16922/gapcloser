# Nebius × NVIDIA Global AI Hackathon — 출품 제안 (v2: Sim2Real)

작성: 2026-10-05 · 마감: **2026-10-30 10:00 PDT (한국시간 10-31 02:00)**, 남은 기간 25일
v1(SpendWarden, MutantGate)은 임팩트가 부족해 접었다. 요약은 맨 아래 부록에 남긴다.


> **v3 갱신 (2026-10-05) — 아래 v2 본문 중 일부는 더 이상 유효하지 않다.** 결정 근거는 `docs/DECISIONS.md`.
>
> - **비용 0, 장비 구매 없음:** Tier 0만 제출한다. Tier 1(SO-101)과 Tier 2(Jetson)는 접는다. §4·§7·§8·§9의 하드웨어와 GPU 비용 항목은 무효다.
> - **실행 위치:** Mac 로컬(M4 Max). 모델은 Nebius Token Factory API로 호출한다(무료 크레딧 $25+$25, 규정 필수 요건 충족). AWS($165.93, GPU 할당량 0)는 선택 사항이다.
> - **시뮬레이터:** Isaac Lab 대신 **NVIDIA Newton**(Mac CPU). 검증 실험을 통과했다: 마찰 격차가 재현되고 카메라 영상이 렌더링된다.
> - **영상 진단:** Token Factory의 Nemotron 3 Nano Omni(영상 입력 확인 필요). 없으면 Cosmos Reason 2(build.nvidia.com이나 AWS).
> - **GR00T, Cosmos Transfer, Isaac Lab:** 선택 사항. 여유와 크레딧이 있을 때만.
> - **핵심 지표:** "숨긴 값 맞히기"가 아니라 같은 예산에서 A 전체 무작위화 / B 공칭값 / GapCloser의 실측 성공률 비교. 현재 오프라인 기준 구현 결과는 19% / 33% / 94%다.
> - 현재 계획: `docs/NEXT_PLAN.md` · 가입 방법: `docs/setup/NEBIUS_SETUP.md`

---

## 0. 한 줄 요약

> **"로봇이 현실에서 실패하면, 에이전트가 실패 영상을 보고 원인을 찾아 시뮬레이터를 고치고, 다시 학습시켜, 다음 시도에서 성공하게 만든다."**

프로젝트 이름(가칭): **GapCloser — 스스로 Sim2Real 격차를 메우는 에이전트**
트랙: **Physical AI**

---

## 1. 왜 Sim2Real인가

### 해커톤이 이 방향을 직접 요구한다

Physical AI 트랙 원문:
> "Build embodied and edge agents that sense and act in the real world … **powered by Nemotron, GROOT, Cosmos and Sonic models and coordinated through an agent runtime.**"

데모 영상 조건 원문:
> "at least a 1-minute clip showing the physical hardware/robot actually operating, **or — if your project has no physical hardware component — the key application modules in action.**"

여기서 알 수 있는 것:
- 스폰서가 원하는 그림은 **Nemotron(두뇌) + GR00T(행동) + Cosmos(눈과 세계 모델)를 에이전트가 조율하는 것**이다. GapCloser가 정확히 이 구조다.
- **시뮬레이션만으로도 출품할 수 있다.** 하드웨어가 없어도 실격은 아니다.
- 이 트랙만 **데모 URL이 필수가 아니다.**
- 트랙 1위 상품이 Jetson Orin Nano다.

### 문제가 누구에게나 바로 와닿는다

- 시뮬레이션에서 100% 성공하는 로봇 정책도 현실에서는 실패한다. 조명, 마찰, 카메라 위치, 물체 질감이 조금씩 다르기 때문이다. 이 차이를 **Sim2Real gap**이라고 부른다.
- 지금은 사람이 영상을 돌려 보면서 "조명 때문인가? 마찰 때문인가?"를 추측하고, 시뮬레이터 설정을 손으로 바꾸고, 다시 학습한다. 며칠씩 걸리는 수작업이다.
- **GapCloser는 이 반복을 에이전트가 대신한다.** 실패 → 진단 → 시뮬레이터 수정 → 재학습 → 재시도.

### 영상 한 장면이 곧 이야기다

"실패하는 로봇 → 에이전트가 영상을 보고 이유를 말함 → 시뮬레이터가 바뀜 → 같은 로봇이 성공"이라는 전후 대비는 설명 없이도 이해된다.

---

## 2. 동작 흐름

```
[현실 로봇 실패 영상]
        │
        ▼
① Cosmos Reason 2 — 영상을 보고 실패를 설명
   "3.2초: 그리퍼가 물체보다 2cm 왼쪽에 접근"
   "물체가 시뮬레이션보다 어둡고 반사가 강함"
        │
        ▼
② Nemotron 3 (Super / Ultra) — 가설을 세우고 실험 계획
   "가설 1: 카메라 위치 오차 → 카메라 위치 무작위화 범위 확대"
   "가설 2: 조명 차이 → 조명 무작위화 + Cosmos Transfer로 실사풍 데이터 생성"
        │
        ▼
③ Isaac Lab (Nebius AI Cloud GPU) — 바뀐 설정으로 시뮬레이션 데이터 생성
   + Cosmos Transfer — 시뮬레이션 영상을 실제처럼 보이게 변환
        │
        ▼
④ GR00T N1.x — 새 데이터로 미세조정
        │
        ▼
⑤ 현실 로봇 재시도 → 성공률 측정 → ①로 돌아가 반복
```

**역할 분담 원칙:** 성공률 같은 수치는 실제로 실행해서 센다. LLM에게 성공률을 추정하게 하지 않는다.

---

## 3. NVIDIA·Nebius 기술 활용

| 구성 요소 | 역할 | 해커톤 요건 |
|---|---|---|
| **Nemotron 3 Super / Ultra** (Token Factory) | 에이전트 두뇌. 가설 수립, 실험 계획, 결과 판단, 보고서 작성 | NVIDIA 오픈 모델 ✅ · Token Factory ✅ |
| **Nemotron 3 Nano** | 실험 로그 요약, 반복 작업 | 등급을 나눠 쓰는 설계 |
| **Cosmos Reason 2** (8B, 오픈) | 실패 영상의 시공간 분석. 타임스탬프와 좌표까지 제시 | 트랙이 명시한 Cosmos |
| **Cosmos Transfer** | 시뮬레이션 영상을 실사풍으로 변환해 학습 데이터 보강 | Cosmos |
| **GR00T N1.x** | 로봇 행동 정책. SO-101 팔에 맞춰 미세조정 | 트랙이 명시한 GROOT |
| **Isaac Lab / Isaac Sim** | 시뮬레이션 환경과 도메인 무작위화 | NVIDIA 표준 Sim2Real 경로 |
| **Nebius AI Cloud GPU** | 시뮬레이션, Cosmos, GR00T 실행 (Mac에서는 Isaac Sim이 안 돌아감) | Nebius AI Cloud ✅ |
| **SO-101 로봇팔** (+ LeRobot) | 현실 로봇. 약 $130~250의 저가 오픈소스 팔 | 하드웨어 영상 |

NVIDIA가 공식 문서에서 이미 "SO-101로 시뮬레이션과 실물 시연 데이터를 모아 GR00T를 학습"하는 경로를 안내하고 있다. 바닥부터 개척하는 길이 아니라 **검증된 경로 위에 에이전트를 얹는 것**이라 25일 안에 해 볼 만하다.

---

## 4. 규모를 3단계로 나눈다

하드웨어 배송이나 GR00T 학습이 막혀도 제출은 할 수 있도록 단계를 나눈다.

| 단계 | 내용 | 영상에서 보여 줄 것 | 하드웨어 |
|---|---|---|---|
| **Tier 0 — 측정 가능한 기본판** | "현실" 대신 **물리 값을 숨겨 둔 두 번째 시뮬레이터**를 쓴다. 마찰·질량·카메라 위치를 몰래 바꿔 두고, 에이전트가 실패 영상만 보고 숨긴 값을 찾아낸다. | 숨긴 값 10종 중 몇 개를 맞혔는지, 몇 번 반복해서 격차를 메웠는지 | 불필요 |
| **Tier 1 — 실물 시연 (목표)** | SO-101 팔로 단순한 집어 옮기기. 실패 → 에이전트 루프 → 성공 | 같은 조건에서 10회 시도 성공률 전후 비교 | SO-101 + 카메라 |
| **Tier 2 — 확장 (여유가 있을 때)** | Jetson에서 정책 실행, 에이전트가 루프를 무인으로 반복 | 무인 반복 타임랩스 | Jetson |

**Tier 0이 숨은 강점이다.** 정답을 아는 상태라 에이전트의 진단 정확도를 숫자로 증명할 수 있다. 화려한 데모만 있는 출품작과 달리 "실제로 맞혔다"는 근거가 생긴다. Tier 1이 실패해도 Tier 0만으로 제출할 수 있다.

---

## 5. 3분 영상 구성안

| 시간 | 장면 |
|---|---|
| 0:00–0:20 | SO-101이 빨간 큐브를 잡으려다 놓친다. 자막: "시뮬레이션에서는 98% 성공했던 정책" |
| 0:20–0:50 | Cosmos Reason이 실패 영상에 타임스탬프별로 표시한다 ("3.2초, 그리퍼 2cm 왼쪽") |
| 0:50–1:20 | Nemotron이 가설 3개를 세우고, Isaac Lab 설정이 화면에서 바뀐다. Token Factory 호출 로그가 보인다 |
| 1:20–1:50 | Cosmos Transfer 변환 전후 영상, GR00T 학습을 빠르게 넘김 |
| 1:50–2:30 | 같은 로봇이 재시도해서 성공한다. 성공률 전후 비교 (실측값) |
| 2:30–3:00 | Tier 0 정확도 표, 아키텍처 한 장, "사람이 며칠 걸리던 일을 에이전트가 한 번의 루프로" |

---

## 6. 심사 기준별 예상

| 기준 | 강점 |
|---|---|
| 기술 구현 | Nemotron, Cosmos 2종, GR00T, Isaac Lab, Nebius GPU를 하나의 루프로 연결. Tier 0의 정확도 수치 |
| 디자인 | 실패 영상 위에 진단을 겹쳐 보여 주는 화면, 반복 타임라인 |
| 잠재 영향 | Sim2Real gap은 로봇 산업 전체의 병목이다. 이 루프 자동화가 곧 개발 속도다 |
| 아이디어 품질 | "로봇을 학습시키는 에이전트"라는 한 단계 위의 시점. 단순한 로봇 제어 데모와 차별화됨 |

솔직한 평가:
- **v1보다 임팩트가 훨씬 크고, 대상권 이야기도 가능하다.**
- 대신 **실패 위험도 훨씬 크다.** 하드웨어, GPU 환경, 학습 시간이 모두 25일 안에 맞물려야 한다.
- Tier 0을 안전망으로 두는 이유가 이것이다.

---

## 7. 위험과 확인할 것

| 위험 | 영향 | 대응 / 확인 시점 |
|---|---|---|
| **SO-101 확보** | 국내 재고와 배송 기간을 모른다. 조립과 보정에도 며칠이 든다 | **10/6에 즉시 주문.** 조립 완제품을 우선 찾는다. 10/15까지 도착하지 않으면 Tier 0으로 제출 |
| **Isaac Lab / GR00T를 Nebius GPU에서 돌리기** | Isaac Sim은 RTX 계열 GPU가 필요하다(L40S 가능성). 컨테이너 설정이 오래 걸릴 수 있다 | **10/8까지 환경 구축을 go/no-go 판정** |
| **GR00T 미세조정 시간과 비용** | 한 번 학습에 몇 시간이 걸리면 반복 횟수가 제한된다 | 처음부터 작은 데이터와 짧은 학습으로 측정한다 |
| **Cosmos Reason이 Token Factory에 있는지** | 목록에서 확인하지 못했다 | 없으면 Nebius GPU에 직접 띄운다(8B라 L40S 1장이면 충분할 가능성) |
| **에이전트 진단이 틀릴 수 있음** | 그럴듯하지만 틀린 원인을 말할 수 있다 | Tier 0의 정답 비교로 정확도를 측정해 공개한다 |
| **현실 성공률이 오르지 않음** | 영상의 결정적 장면이 사라진다 | 쉬운 과제(단색 큐브 집기)로 범위를 좁힌다. 오르지 않으면 정직하게 보고하고 Tier 0을 중심으로 제출 |

---

## 8. 비용 예상 (확인 필요)

| 항목 | 예상 | 근거 |
|---|---|---|
| SO-101 키트 + 카메라 | 약 $150~300 | 공개 키트 가격대. 국내 배송비 별도 |
| Nebius GPU — L40S | $1.55/시간 | 가격 비교 사이트 기준, 공식 확인 필요 |
| Nebius GPU — H100 | $2.95~3.5/시간 | 같은 기준 |
| GPU 총사용 (100~150시간 가정) | 약 $200~400 | 시뮬레이션 + 학습 + Cosmos |
| Token Factory | 크레딧 $50 안에서 해결 | Nemotron 호출 위주 |
| **합계** | **약 $400~700** | GPU는 반드시 자동 종료와 예산 경보를 먼저 설정 |

---

## 9. 일정 (KST)

| 기간 | 할 일 | 끝났다는 기준 |
|---|---|---|
| 10/5–10/8 | SO-101 주문 · 크레딧 신청 · Nebius GPU에 Isaac Lab과 GR00T 설치 · Cosmos Reason 실행 · Token Factory Nemotron 에이전트 뼈대 · **go/no-go** | 시뮬레이션에서 GR00T가 한 동작을 실행함 |
| 10/9–10/15 | **Tier 0 완성:** 물리 값을 숨긴 시뮬레이터 · 에이전트 진단 루프 · 정확도 측정 | 숨긴 값 10종에 대한 정답률 표 |
| 10/16–10/22 | 로봇 조립·보정 · 시연 데이터 수집 · 현실 기준 성공률 측정 · Cosmos Transfer 연결 | 현실에서 10회 시도한 기준 성공률 |
| 10/23–10/27 | **Tier 1 루프:** 현실 실패 → 진단 → 재학습 → 재시도 | 개선 전후 성공률 (실측) |
| 10/28–10/29 | 영상 · README · LICENSE · Devpost 본문 · 피드백 | **10/29 제출** |

---

## 10. 정해 주셔야 할 것

1. **이 방향(Sim2Real, GapCloser)으로 갈지**
2. **하드웨어:** SO-101 등 로봇팔이나 Jetson을 이미 갖고 있는지. 없다면 새로 사서(약 $150~300) Tier 1까지 갈지, Tier 0만으로 갈지
3. **GPU 예산:** Nebius AI Cloud에 약 $400~700을 쓸 수 있는지. 상한액을 정해 주시면 그 안에서 계획한다
4. **작업 시간:** 남은 25일 동안 하루에 쓸 수 있는 시간. Tier 1의 로봇 조립과 데이터 수집은 사람이 직접 해야 한다

---

## 11. 확인하지 못한 것

- Cosmos Reason 2와 Cosmos Transfer를 Token Factory에서 바로 쓸 수 있는지
- "Sonic" 모델의 정체 (트랙 문구에는 나오지만 확인하지 못함)
- Nebius에서 Isaac Sim이 동작하는 GPU 종류와 공식 단가
- SO-101 국내 구매처와 배송 기간
- GR00T 미세조정에 실제로 걸리는 시간과 필요한 데이터 양

---

## 부록: 접은 v1 후보

- **SpendWarden** (Personal AI): 개인 클라우드 청구 감시 에이전트. 실용적이지만 와닿는 정도와 주목도가 약하다는 판단으로 접음.
- **MutantGate** (Coding and Agentic): 변이 테스트로 자기 검증하는 코딩 에이전트. 출품이 가장 몰릴 트랙이고 시각적 임팩트가 약함.

## 출처

- [해커톤 메인](https://nebiusglobalaihackathon.devpost.com/) · [규정](https://nebiusglobalaihackathon.devpost.com/rules) · [자료](https://nebiusglobalaihackathon.devpost.com/resources) · [일정](https://nebiusglobalaihackathon.devpost.com/details/dates)
- [GR00T N1.6 Sim-to-Real 워크플로](https://developer.nvidia.com/blog/building-generalist-humanoid-capabilities-with-nvidia-isaac-GR00T-n1-6-using-a-sim-to-real-workflow) · [Isaac Teleop: LeRobot and SO-101](https://nvidia.github.io/IsaacTeleop/main/getting_started/lerobot/index.html)
- [Cosmos Reason 2 모델 카드](https://build.nvidia.com/nvidia/cosmos-reason2-8b/modelcard) · [Cosmos Reason 2 문서](https://docs.nvidia.com/cosmos/latest/reason2/)
- [Nebius AI Cloud 가격](https://nebius.com/prices) · [L40S 가격 비교](https://computeprices.com/providers/nebius/gpus/l40s)
- [Nebius Token Factory × Nemotron](https://nebius.com/services/token-factory/nemotron)
