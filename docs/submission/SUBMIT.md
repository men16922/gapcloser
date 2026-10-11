# 제출 안내: 올리기만 하면 됩니다

마감: **2026-10-30 10:00 PDT (한국 시간 10-31 02:00)**. 여유 있게 10-29까지 제출하는 것을 권합니다.

모든 파일은 저장소의 `submission/` 폴더에 있습니다. 이 폴더는 git에 올라가지 않습니다. 다시 만들려면 `make submission`을 실행합니다.

| 파일 | 쓰는 곳 |
|---|---|
| `tether_film.mp4` | YouTube 업로드 (2분 28초, 1080p) |
| `youtube-thumbnail.png` | YouTube 미리보기 이미지 (1280×720) |
| `youtube-description.txt` | YouTube 설명란 (챕터 포함) |
| `gallery/00-cover.png` | Devpost 프로젝트 대표 이미지 (3:2) |
| `gallery/01~09-*.png` | Devpost 이미지 갤러리 (3:2) |
| `about-the-project.md` | Devpost "About the project" 본문 |

해커톤 영상 조건은 다음과 같습니다.

- 공개 YouTube 영상이어야 하고 3분 이내여야 합니다. 이 영상은 2분 28초입니다.
- 내레이션이 Nebius Token Factory와 Nemotron을 어떻게 썼는지 말해야 합니다. 0분 40초 부근과 마지막 장면에서 말합니다.
- 하드웨어가 없는 Physical AI 프로젝트는 앱이 실제로 동작하는 화면을 1분 이상 보여야 합니다. 이 영상에는 Studio를 실제로 조작하는 화면이 약 68초 들어 있습니다.

---

## 1단계. YouTube 업로드 (약 10분)

1. https://studio.youtube.com → 만들기 → 동영상 업로드 → `submission/tether_film.mp4`
2. 제목 (그대로 붙여 넣기):
   ```
   Tether: tie your simulator to the real world (NVIDIA Newton + Nemotron on Nebius Token Factory)
   ```
3. 설명: `submission/youtube-description.txt` 내용을 전부 붙여 넣습니다.
4. 미리보기 이미지: `submission/youtube-thumbnail.png`. 계정 전화번호 인증이 안 되어 있으면 이 항목이 막힙니다. 그때는 건너뛰어도 됩니다.
5. 시청자층: "아니요, 아동용이 아닙니다"
6. 공개 상태: **공개** (해커톤 규정상 공개 영상이어야 합니다)
7. 업로드가 끝나면 영상 주소(`https://youtu.be/...`)를 복사해 둡니다.

## 2단계. Devpost 제출 (약 15분)

https://nebiusglobalaihackathon.devpost.com 에서 "Enter a submission" (또는 "Submit project")을 누릅니다.

| Devpost 항목 | 넣을 내용 |
|---|---|
| Project name | `Tether` |
| Elevator pitch (200자 이내) | 아래 A |
| Thumbnail | `submission/gallery/00-cover.png` |
| About the project | `submission/about-the-project.md` 전체 (마크다운 그대로) |
| Built with | 아래 B (태그를 하나씩 입력) |
| "Try it out" links | `https://tether-454741001655.us-central1.run.app` 와 `https://github.com/men16922/tether` |
| Image gallery | `submission/gallery/01~09` 이미지 |
| Video demo link | 1단계에서 복사한 YouTube 주소 |
| Track / Category | **Physical AI** |
| 팀원 | 본인 (팀원이 있으면 초대) |

추가 질문이 나오면 아래 C의 답을 씁니다.

**A. Elevator pitch**
```
Tether ties your simulator to the real world: from a phone video or a robot log, Nemotron finds the physics your sim gets wrong, NVIDIA Newton checks the fix and retrains the policy.
```

**B. Built with**
```
nvidia-newton, nvidia-warp, nemotron, nebius-token-factory, nemo-agent-toolkit, cosmos-reason, mujoco, python, fastapi, three.js, opencv, scipy, google-cloud-run, docker
```

**C. 자주 나오는 추가 질문 (영문 답)**

- How does the project use Nebius?
  ```
  The reasoning agent runs NVIDIA Nemotron 3 Super (nvidia/nemotron-3-super-120b-a12b) on Nebius Token Factory through its OpenAI-compatible API with tool calling: it profiles deceleration, checks perception, proposes model structures and asks for the runs that would settle them. The "Ask Tether" chat on the live demo also runs on Token Factory.
  ```
- Which NVIDIA open models / technologies?
  ```
  NVIDIA Nemotron 3 Super (agent and chat, on Nebius Token Factory); NVIDIA Newton and Warp (physics, rendering, replay check, retraining in 16 parallel worlds); NVIDIA NeMo Agent Toolkit (the calibration agent as a NAT workflow); NVIDIA Cosmos Reason 2 (optional local second opinion on clips).
  ```
- How can judges test it?
  ```
  Open https://tether-454741001655.us-central1.run.app/studio (the first visit after an idle spell takes ~15 s to start). Pick a field (e.g. Autonomous vehicles), "Open an example", choose the roadside camera, press "Track pushes", read Nemotron's notebook, then "See the calibrated simulator". "Simulate your own" lets you hide the physics yourself and check whether Tether finds it ("Reveal the hidden physics"). "Ask" answers questions about your own result. Everything runs on CPU; retraining takes about 3 minutes on the server.
  ```
- Hardware? (Physical AI 트랙)
  ```
  No physical hardware. The video shows the application modules in action for over a minute (Studio tracking, the Nemotron agent, the calibrated simulator, Newton replay and export, Ask Tether), plus retraining in parallel NVIDIA Newton worlds and validation on real objects from the EV-RealPhys benchmark.
  ```

## 3단계. 제출 전 마지막 확인 (5분)

- [ ] 공개 데모 주소가 열리는지 확인합니다. 처음 접속은 15초쯤 걸릴 수 있습니다.
- [ ] GitHub 저장소가 공개 상태이고, README 맨 위에 데모 주소가 있는지 확인합니다.
- [ ] YouTube 영상이 "공개"이고 로그아웃 상태에서도 재생되는지 확인합니다.
- [ ] Devpost 미리보기에서 영상과 이미지가 보이는지 확인한 뒤 **Submit**을 누릅니다.

## 제출 후

- 심사가 끝날 때까지 Cloud Run 서비스를 그대로 둡니다. 방문자가 없으면 0대로 꺼지므로 비용은 거의 없습니다.
- 심사가 끝난 뒤 내리려면 다음 명령을 실행합니다.
  ```bash
  gcloud run services delete tether --region us-central1 --project claude-study-501117
  ```

## 별도 기술 글 (선택)

`docs/article/`에 한국어(`tether.ko.md`)와 영어(`tether.en.md`) 기술 글이 있습니다. 이미지는 `docs/article/img/`에 있습니다.

1. YouTube에 올린 뒤, 두 파일의 `YOUTUBE_LINK`를 영상 주소로 바꿉니다.
2. 글 전체를 마크다운 편집기(velog, dev.to 등)에 붙여 넣습니다. 이미지는 GitHub 저장소의 `docs/article/img/`에서 바로 불러오므로 따로 올리지 않아도 됩니다.
3. 대표 이미지(커버)를 따로 지정하는 곳이면 `docs/article/img/cover.jpg`를 올립니다.
4. 추천 태그: `nvidia`, `robotics`, `simulation`, `llm` (한국어: 로봇, 시뮬레이션, LLM, NVIDIA)
