# Nebius 사용 내역, 비용, 해커톤 요건

## 해커톤 필수 요건

- 요건은 "Nebius Token Factory **또는** Nebius AI Cloud에서 동작 + NVIDIA 오픈 모델 1개 이상 사용"입니다 (`docs/setup/NEBIUS_SETUP.md`).
- Tether는 Nebius Token Factory에서 NVIDIA Nemotron 3 Super를 호출하므로 Token Factory만으로 요건을 충족합니다.
- Nebius AI Cloud(VM)는 필수가 아닙니다. 공개 데모 주소가 필요할 때만 선택적으로 사용합니다.

## Nebius AI Cloud 사용 내역 (2026-10-09)

| 항목 | 내용 |
|---|---|
| 결제 정보 등록 | 사용자가 등록, $25 선충전 |
| VM | `gapcloser-demo`, eu-west1, cpu-d3 2 vCPU / 8 GB, 디스크 40 GB |
| 사용 시간 | 약 5.5시간 (배포 확인 후 요청에 따라 삭제) |
| 확인용 리소스 | 디스크 1개, VM 1개를 실수로 생성 후 즉시 삭제 |
| **총 사용액** | **$0.38** (CPU $0.17, RAM $0.20, 디스크 $0.02) |
| **잔액** | **$24.62** |
| 현재 리소스 | 전 리전 VM 0, 디스크 0, 공인 IP 0, 저장소 0 (추가 과금 없음) |

출처: Nebius 콘솔 Billing → Usage (2026-10-09 12:22 UTC 기준), `nebius` CLI 조회.

## 환불 문의

- 남은 $24.62는 선충전 잔액입니다. 자동으로 환불되지 않으며, 사용하지 않으면 계정에 남습니다.
- 환불은 Nebius 지원팀에 직접 요청해야 합니다. 콘솔 오른쪽 위 도움말(?) 메뉴의 지원 요청, 또는 Billing 화면의 "Send billing feedback"을 이용합니다.
- 요청 내용 예시 (영문):

  > Hello. I added billing details to tenant crimson-silverfish-tenant-abc (customer ID customer-e00ehudcbgahcv5avj8il) and was charged a $25 top-up. I used $0.38 for a short test and have deleted all resources. I only need Nebius Token Factory for the hackathon. Could you refund the unused balance of $24.62? Thank you.

- 환불 가능 여부와 기간은 Nebius 정책에 따릅니다. 가능 여부는 확인되지 않았습니다.
- **2026-10-10 00:16 환불 요청 제출:** 티켓 P282039303 (Billing → Refunds, "Refund request for unused prepaid balance ($24.62)", 상태 Open). 진행 상황은 콘솔 Support center에서 확인합니다.

## 공개 배포

공개 데모는 Google Cloud Run에서 운영합니다(`make deploy`, 자세한 내용은 `docs/deploy/DEPLOY.md`). Nebius VM 배포는 쓰지 않습니다.

## Token Factory 사용량

- Token Factory 크레딧에서 차감되며, 카드로 청구되지 않습니다.
- 10-08 기준 사용액은 약 $2였고, 10-09에 녹화·테스트·질문으로 추가 사용했습니다(추정 $1 미만). 질문 1회는 약 $0.002, AI 진단 1회는 약 $0.01입니다.
- 정확한 잔액은 https://tokenfactory.nebius.com 에서 확인합니다.
