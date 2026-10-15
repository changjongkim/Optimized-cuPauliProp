# SectorQ 진행 상태 — 2026-10-15 00:50 UTC

이 파일은 `2026-10-14_sectorq_handoff.md`의 후속 기록이다. 원래 0절 규칙을 유지한다. 모든 쓰기는 kcj 안에서, 우리 `pp_` job만, 새 8노드 이상 job은 사용자 승인 뒤, 할당 하나에 loop 하나. push는 요청이 있을 때만 한다.

## 이번 턴의 결과

- 인계된 변경을 먼저 커밋했다: `15b6a6f`(승인된 설계 그림/절 구성), `0c6f3bc`(고정 행 실험 재개기).
- 설계 3.1–3.5, 확보된 결과를 사용한 평가, 관련 연구 본문을 작성했다. 설계 개념 그림은 수정하지 않았다.
- 고정 행 비교, 1–512 GPU scale-out, 원래 회로 기준 128 GPU 비교, 128–512 GPU 시간 분해 그림을 추가하고 weak 결과를 갱신했다.
- 서론 수정 초안을 먼저 제시한 뒤 인용, ORQA 분배 함수 설명, GPU 구현에 대한 주장 범위, 메모리 산식을 정정했다. 사용자가 승인한 `In this paper` 문단은 바이트 단위 비교로 유지됨을 확인했다.
- 인용 TODO와 서지정보의 미확인 TODO를 제거했다. Atlas, Häner–Steiger, tapering, commutator graph, iQCC, paulistrings-rs, ORQA, MonoProp, Julia, SPD, Qiskit, cuPauliProp, PADO-Pauli, Backpropagating PP의 원문/공식 저장소를 확인했다.
- `bash paper/scripts/build.sh` 성공, `python3 paper/scripts/check_style.py` 0건. 26쪽, 검사기의 본문 문단 합계 9,467단어(초록 169, 승인 서론 마지막 문단 190, 결론 106). 미해결 인용/참조와 Overfull 없음. acmart의 기존 그림 description/익명 소속 경고와 일부 Underfull, 학회 서지의 address 경고는 남는다.
- PDF: `/pscratch/sd/s/sgkim/kcj/pauliprop/paper/main.pdf`. GitHub 링크는 https://github.com/changjongkim/SectorQ/blob/main/paper/main.pdf 이며 **push하지 않았으므로 이번 수정은 아직 없다**.

## 실험 상태와 재개

### N2 균형 선택 ablation — 실행 전

`jobs/b_ablate.sh`를 만들었다. 기본 4노드/16 GPU, 같은 `pp_engine_x`, 8개 후보 대 1개 후보, trace 기록, 일곱 회로다. `--cands 1`은 후보 간 비교만 없애며, 표본 수집과 sector 수 선택까지 없애는 것은 아니다.

한 차례 제출이 `QOSMaxSubmitJobPerUserLimit`로 거절되었고 job은 만들어지지 않았다. `logs/b/loop.log`에 오류가 있다. 기다리는 loop는 없다. 공유 계정의 다른 작업을 중지하거나 재사용하지 않았다.

재개 전 `logs/b/loop.pid`, 해당 `/proc/<pid>/cmdline`, `timeout 15 pgrep -u sgkim -f 'jobs/b_ablate.sh'`, 할당의 active step을 확인한다. 오래된 PID가 다른 프로세스에 재사용되었으면 그 프로세스를 건드리지 않는다. 실행기는 campaign flock과 allocation flock을 사용하며 요청은 `--immediate=30`으로 한 번만 한다. 재시도 대기 loop는 없다. 이후 코드에는 종료 시 직접 만든 `pp_b` 할당만 반납하는 trap도 추가했다.

다음 턴에 interactive 제출 여유가 있으면 중복 프로세스가 없음을 확인한 뒤 다음과 같이 분리 실행한다. 루트 build.sh와 대기 job의 바이너리는 다시 빌드하지 않는다.

```python
import subprocess
with open('logs/b/loop.log', 'ab', buffering=0) as log:
    subprocess.Popen(['nohup', 'setsid', 'bash', 'jobs/b_ablate.sh'],
                     stdin=subprocess.DEVNULL, stdout=log,
                     stderr=subprocess.STDOUT, start_new_session=True)
```

`PP_B_JOB`는 idle 상태의 4노드 `pp_` 할당에만 사용한다. 다른 campaign loop가 그 할당을 다시 사용할 가능성이 없어야 한다. 실행 중 기다리지 말고 확보된 결과를 보고한다.

결과가 생기면 wall time, 기대값, GPU/host split, 노출된 row wait, 구간별 가장 큰 sector/평균을 비교한다. trace의 `sectors`는 **비어 있지 않은 sector 수**이다. 평균은 nonempty 기준과 전체 `2**rows` 기준을 구별한다. `analysis/paper_ablation.py`의 `parse_log()`가 두 불균형을 계산한다. N2 효과가 작거나 음수이면 먼저 사용자에게 알리고 독립적인 성능 주장을 낮춘다.

### 대기 중인 job — 변동 없음

00:50 기준 모두 PENDING(Priority):

| Job | Name | Nodes |
|---|---|---:|
| 59768854 | pp_s_g256w | 64 |
| 59768995 | pp_s_g32w | 8 |
| 59768996 | pp_s_g64w | 16 |
| 59768998 | pp_s_g128w | 32 |
| 59768999 | pp_s_g256w2 | 64 |
| 59769000 | pp_s_g512w | 128 |
| 59769001 | pp_p_g32 | 8 |
| 59769002 | pp_p_g64 | 16 |

남은 반영 대상은 N2, CPU 추가 ladder/weak/reach, paulistrings-rs 32/64 GPU, 재측정 중인 256 GPU 한계다. 평가 파일의 `% TBD` 주석 다섯 곳에 위치를 남겼다. 원고와 실험 전체가 완료되었다고 보고하면 안 된다.

## 새로 확인한 근거와 주의점

1. `queue_s/todo_a0o_orig.jsonl`의 원래 회로 cuPauliProp 66건은 인계문과 달리 이미 모두 terminal 상태다. 성공 41, OOM 14, 실패 이후 생략 11. 128 GPU 원래 회로 KI/noise 결과를 사용할 수 있어 새 그림에 적용했다. 128 GPU 최대 공통 완료 설정에서 SectorQ 7.4–13.4배. TFIM은 원래 입력과 frame 입력이 같다.
2. 고정 행 ablation은 `logs/x/*.out` 중 정확한 `re|fx.<circuit>.g4|g16.<cutoff>.out`만 사용한다. 하위 contended/old_buggy/alloc_end, `.t1` 등은 제외했다. 성공 쌍은 16 GPU 25개, 4 GPU 22개이며 출력 기대값은 모두 같다. 서로 window 경계가 다르므로 **최대 boundary 항 수는 두 모드에서 같을 필요가 없다**. 그림 x 항 수는 recomputed 모드 기준이다.
3. 16 GPU 고정/재계산 시간 비율 2.43–17.03. KI 1e-5는 137.069/11.938초, 878/20구간, 698.507/45.221 GB 전송. 기존 element sequence의 SectorQ 12.2초와 별도 할당이므로 하나로 합치지 않았다.
4. 기본 gmax=8은 512 GPU에서도 boundary sector 상한 256이다. KI 3e-6은 trace에서 최대 256개 nonempty sector와 split 0, 256→512 GPU wall 10.478→9.669초, idle worker 비중 9.0→40.2%. TFIM/noise에서도 유사하다. 이 설정의 효율 저하를 네트워크 하나로 설명하면 안 된다.
5. 상한만 늘리면 해결된다고도 쓰지 않는다. 기존 `n12.ki.3e-6.g512`는 gmax=12와 chunk=65536을 함께 사용했고 11.568초로 기본 9.669초보다 느리다. 이는 두 파라미터의 공동 변경이라 cap 단독 ablation은 아니다.
6. 최대 512 GPU 실행 `i12.ki.1.5e-7.g512`는 3,413.465초, boundary 448.2B, 17.9 TB payload, host chunk pool 21,079 GiB(설치 host의 64%). library operations 46%, exchange 26%, GPU-host copies 약 9%, idle 0.5%. exchange 중 transfer 607초, posting 280초, counts 2초 미만. 잔여 시간을 전부 planning으로 부르지 않는다.
7. worker의 operations/idle 등은 **누적 worker 시간**이며 GPU utilization 샘플이 아니다. worker 수로 나눈 평균과 노드 평균 exchange/sync를 wall과 비교했다. `analysis/paper_large_time.py` 참조.
8. `max_terms_inside_a_window`는 각 sector 실행의 최대 항 수를 더한 값이다. 동시에 메모리에 존재하는 전역 최대라고 해석하면 안 된다. 서론은 실제 boundary 159M(1e-5), 11.856B(1e-6), 40 B/term으로 약 6.4/475 GB를 사용한다. 배경은 내부 합산 counter의 정의를 명시했다. 기존 `4 TB = GPU 1,000장` 문장은 제거했다.
9. KI 낮은 cutoff의 기대값은 단조 수렴하지 않는다(1e-6부터 약 0.3217, 0.3256, 0.3265, 0.3226, 0.3153, 0.3089). 정확한 물리 오차가 반드시 작아졌다는 주장과, 동일 truncation의 더 큰 실행을 가능하게 했다는 주장을 구분했다.

## 재생성 및 검증

```bash
python3 analysis/paper_ablation.py
python3 analysis/paper_large_time.py
cd figures
../env/bin/python3 make_paper_figures.py ev_fixed_rows ev_large_time ev_scaleout ev_circuits128 ev_weak
cd ..
bash paper/scripts/build.sh
python3 paper/scripts/check_style.py
```

원본 로그는 기존 경로에 유지했다. 분석 JSON/CSV에는 상대 로그 경로가 있다. 고정 행 JSON은 summary만 저장하며 trace가 필요하면 parse_log()로 원본을 읽는다. 큰 규모 JSON에는 trace도 있다. 추가 그림만 생성하고 설계 그림 생성기는 실행하지 않는다.

## 사용자의 novelty 질문에 대한 답변 방향

현재 확인한 선행 연구 기준으로 N1이 시스템 기여의 중심이다. parity/선형 분할 자체, stage 실행 자체, host offload 자체가 새롭다고 주장하지 않는다. 구간의 모든 생성자로부터 merge-local partition을 구성하고, 살아 있는 항에 맞춰 재선택하며, 분할 가능한 sector를 host memory에서 GPU로 실행하는 메커니즘이 핵심이다. 고정 행 ablation은 N1의 실질적 효과를 뒷받침한다. N2의 후보 선택 효과는 실험 전이므로 독립적으로 강하다고 단정하지 않았다. 관련 연구 본문은 이 구분에 맞췄다.

## Git 및 미완료 의사결정

코드와 논문 커밋을 분리하고 경로를 명시했다. 논문 커밋에는 Co-Authored-By를 넣지 않는다. push하지 않는다. 다른 작업의 staged `queue_f/ext.log`, `queue_f/runner.log`, untracked `jobs/e14_scaleout.sh`, compare 그림과 로그는 손대지 않았다.

공개 저장소 익명화와 승인된 서론 마지막의 anonymous URL 유효성은 아직 사용자 결정/확인이 필요하다. 새 8노드 이상 실험이 필요해지면 결과와 구체적 계획을 준비한 뒤 승인받는다.

## 후속 변경 — Evaluation 통합 (2026-10-15)

- 사용자의 후속 요청으로 `af8e176`까지 GitHub main에 push를 완료했다. 위의 push 전 상태 설명은 00:50 시점 기록이다.
- 이후 사용자가 Evaluation의 분리된 결과를 합칠 수 있는지 점검해 달라고 요청했다. 기존 11개 소절, 15개 그림, 24개 제목 문단을 6개 소절, 12개 그림, 19개 제목 문단으로 재구성했다.
- 새 순서는 Setup → Performance Compared with Existing Systems → Scalability and Capacity → Contribution of the Design Choices → Numerical Agreement and Accuracy → Parameter Sensitivity and Repeatability다.
- 성능 비교에 single-node, 16 GPU, 128 GPU, paulistrings-rs를 모았다. paulistrings-rs는 truncation 주기가 다르므로 cuPauliProp 비교 곡선과 합치지 않았다.
- 항당 시간 7개 패널은 실행 시간/항 수의 재표현이므로 본문에서 중복 그림을 제거하고 지표 정의와 수치를 16 GPU 성능 문단에 통합했다. 원본 ev_cost PDF와 생성 코드는 저장소에 그대로 있다.
- Strong/weak scaling, 회로 크기/깊이, 최대 operator/host 사용량을 한 소절로 모았다. 회로 크기/깊이는 한 문단, 최대 항 수/host 사용량도 한 문단으로 통합했다.
- 고정 행 ablation과 요소별 계단, 시간 분해를 같은 소절에 배치했다. KI의 구간 수/전송량 중복 설명을 제거했다. N2가 완료되면 이 소절 안에 추가하며 새 독립 소절을 만들지 않는다.
- 16 GPU 및 128–512 GPU 시간 분해는 한 그림의 6개 패널로 합쳤다. 동일한 색상 범주를 확인하고 기존 large-time 범례를 공통으로 썼다. 퍼센트/초 축의 차이는 캡션에 명시했다.
- Parameter sensitivity와 반복 측정은 한 소절 및 한 그림으로 묶되, 실험 조건과 범례를 각 패널 묶음에 유지했다. 구현 간 답 일치는 Accuracy로 옮겼다.
- 새 데이터/설계 그림은 그리지 않았다. 기존 그림 PDF 83개가 모두 수정 전 SHA-256과 같음을 확인했다. 합친 그림을 PDF 19/21쪽에서 렌더링해 확인했다.
- 결과: Evaluation 본문 3,601 → 2,757단어, 전체 본문 9,467 → 8,623단어, 원고 26 → 24쪽. 컴파일 성공, check_style 0건, 미해결 인용/참조와 Overfull 없음. 대기 실험의 TBD 주석 다섯 곳은 유지했다.
- 이 구조 변경은 로컬 커밋으로 남긴다. 앞선 push 요청은 이전 버전에 대해 완료했으며 이번 구조 변경은 아직 push하지 않았다.

## 후속 변경 — Figure 17의 2×3 배치 (2026-10-15)

- 사용자의 추가 push 요청으로 Evaluation 통합 커밋 `9c7af6b`를 GitHub main에 push했다.
- 사용자가 Figure 16 유지, Figure 17만 수정으로 명확히 지정했다. 앞서 작업 중이던 Figure 16 삭제/표 압축과 Background 수정은 되돌렸다. Figure 16과 Table 6, Background는 `9c7af6b`와 같다.
- Figure 17의 (a)–(c)를 가로 누적 막대에서 세로 누적 막대로 바꾸고, 아래 128–512 GPU 패널과 같은 크기·색상·글자 크기를 사용했다. 위 3개, 아래 3개와 공통 범례로 배치했다.
- 위쪽은 16 GPU의 기존 다섯 cutoff별 시간 비중(%), 아래쪽은 기존 128/256/512 GPU 실행 시간(s)이다. 기존 `g_breakdown_sizes`가 반환하는 동일 측정값을 재사용했다. 새 실험이나 데이터 변경은 없다.
- `figures/make_paper_figures.py ev_breakdown`만 실행했다. 데이터 패널 3개와 해당 범례만 갱신했고, 설계 그림과 나머지 그림 PDF는 SHA-256으로 동일함을 확인했다.
- 컴파일 성공, check_style 0건, 미해결 인용/참조와 Overfull 없음. PDF 19쪽을 렌더링해 2×3 배치와 다섯 cutoff의 가독성을 확인했다. 본문 8,623단어(Evaluation 2,757단어)는 그대로이며 원고는 24 → 23쪽이다.
- 이번 Figure 17 변경은 로컬 커밋만 한다. 새로운 push 요청은 아직 없다. 실험 대기 상태는 이 편집 작업에서 다시 조회하지 않았다.

## 후속 변경 — 데이터 그림의 폭과 글자 정리 (2026-10-15)

- Figure 17 수정 후 사용자의 push 요청에 따라 `9060b77`까지 GitHub main에 push했다.
- 사용자가 Figure 9·11의 가로폭/여백 개선을 요청했고, 이어서 모든 데이터 그림의 축 외 굵은 글씨 제거, 작은 폰트, 데이터 글자 넘침 해결로 범위를 넓혔다. 설계 그림은 그대로 유지했다.
- Figure 9·11은 4+3 배치를 유지하면서 행마다 가로폭을 채운다. 같은 행의 실제 플롯 폭을 맞추고, 동일한 로그 y축의 눈금/제목은 행의 첫 패널에만 둔다. 옅은 가로 격자와 일관된 배수 표시를 사용한다. Figure 11의 범례에도 배수의 정의를 넣었다.
- 모든 데이터 패널/범례를 재생성했다. 축 제목만 bold, 눈금과 범례는 11 pt, 데이터 표시는 10 pt, 축 제목은 11.5 pt다. 본문 삽입 배율 0.6 기준 6.6/6.0/6.9 pt로, 기존 눈금·범례 12 pt와 수치 11 pt보다 약 8–9% 작다.
- 데이터 글자는 기준점을 유지하고 텍스트 위치를 조정해 플롯 내부에 둔다. 축 모서리의 눈금 충돌은 간격을 확보해 해결한다. 긴 Background y축 제목은 두 줄로 나누고, 해당 설명과 메모리 민감도 끝점 숫자 위에 여백을 확보했다. Figure 12의 범례도 공통 0.6 배율을 사용한다.
- 원본 측정값·분석 파일은 수정하지 않았다. 데이터 PDF 78개만 갱신되었고, 설계/연산 그림 5개는 수정 전 SHA-256과 같다. 별도로 보관하던 항당 시간 그림 생성은 `ev_cost`로 분리했으며 전체 생성 시에는 계속 포함된다.
- 생성기의 글자/범례/데이터 경계 및 겹침 검사 0건. 데이터 그림이 있는 PDF 4–6, 12–21쪽을 모두 렌더링해 확인했다. 컴파일 성공, check_style 0건, 미해결 인용/참조 및 Overfull 없음. 본문 8,623단어(Evaluation 2,757단어), 23쪽이다.
- 재생성은 `figures/`에서 `MPLCONFIGDIR=/pscratch/sd/s/sgkim/kcj/pauliprop/paper/build/matplotlib ../env/bin/python3 make_paper_figures.py`로 한다. 검사 로그와 미리보기는 git 제외인 `paper/build/`에 있다.
- 이번 변경은 로컬 커밋만 하며 새 push 요청은 아직 없다. 이 편집 작업에서는 실험/job 상태를 다시 조회하지 않았다.

## 후속 변경 — Figure 17의 cutoff별 3×3 배치 (2026-10-15)

- 앞선 데이터 그림 정리 후 사용자의 push 요청으로 `6fcb59c`까지 GitHub main에 push했다.
- 사용자가 Figure 17도 cutoff별로 보여 달라고 요청했고, GPU 수는 16·64·256의 세 행으로 지정했다. 열은 kicked Ising, TFIM heavy-hex, noisy kicked Ising이다. 모든 패널을 시간 비중(%)과 공통 로그 cutoff 축으로 통일했다.
- 새 `analysis/paper_time_sweeps.py`가 기존 완료 로그 103개를 JSON/CSV로 정리한다. 16 GPU는 기존 그림과 동일한 15개 실행, 64/128/256/512 GPU는 기본 cap=256, truncEvery=10, 회로별 동일 target의 88개 실행이다. 그림에는 16/64/256 GPU의 59개만 사용하고 128/512 GPU도 분석 파일에 보관했다. 새 실험은 없다.
- 16 GPU의 1e-4/3e-5/1e-5는 `logs/g/eng.*`의 `pp_engine_h`, 3e-6/1e-6은 `logs/r/eng.*`의 `pp_engine_i`로 기존 출처를 유지했다. 64 GPU 이상은 `logs/s/i.ref.ki.*`와 `logs/s/n.eng.*`의 `pp_engine_i`다. 같은 바이너리 버전으로 모두 재측정한 sweep이라고 주장하지 않는다.
- higher-cap/chunk KI 실행은 비교 조건이 다르므로 새 cutoff 그림에서 제외했다. 기존 `paper_large_time.*`와 최대 용량 결과는 그대로다. worker counter/GPU 수와 node-average exchange/sync를 합산하며, Other는 wall에서 남는 시간이다. 103개 모두 음수 성분이 없고 합이 wall과 일치한다.
- 본문에 64 GPU의 idle worker 비중이 cutoff 1e-4에서 41–48%, 회로별 최저 표시 cutoff에서 1–2%라는 결과를 추가했다. 기존 512 GPU의 idle 40%는 별도 측정임을 명시했다.
- `figures/make_paper_figures.py ev_runtime`으로 새 패널 9개와 공통 범례만 생성했다. 기존 PDF 83개는 SHA-256이 모두 같아 Figure 16과 설계 그림을 포함한 다른 그림은 바뀌지 않았다.
- 컴파일 성공, check_style 0건, 미해결 인용/참조와 Overfull 없음. 19쪽을 렌더링해 배치와 가독성을 확인했다. 23쪽, 본문 8,647단어(Evaluation 2,781단어)다. 미리보기는 `paper/build/runtime-cutoffs.png`에 있다.
- 새 변경은 코드/논문 커밋을 분리해 로컬에 남긴다. 이번 변경에 대한 새 push 요청은 아직 없다. 이 편집 작업에서는 실험/job 상태를 다시 조회하지 않았다.

## 후속 변경 — 대표 cutoff, 축 표기, 요구사항 표 제거 (2026-10-15)

- 앞선 Figure 17 변경은 사용자 요청으로 `daecc96`까지 GitHub main에 push했다. 아래 변경에 대한 새 push 요청은 아직 없다.
- 사용자가 Figure 17을 cutoff 1e-4/1e-5/1e-6만 보이도록 줄이고, 빠진 결과가 대기 실험으로 채워지면 16/64/256 GPU 세 행을 유지하도록 요청했다. 세 회로의 세 점이 현재 엔진/기본 상한으로 모두 완료된 규모는 16 GPU다. 64/128/256/512 GPU의 TFIM/noise는 세 점이 있으나 KI의 1e-6은 다른 엔진 또는 상한을 바꾼 실행만 있었다.
- 기존 대기 목록에는 KI 1e-6의 engine i / cap 256 측정이 없었다. 기존 `pp_s_g64w`(59768996, 16노드), `pp_s_g256w2`(59768999, 64노드)의 script가 시작 시 `queue_s/todo*.jsonl`을 읽는 것을 확인했다. 두 job은 확인 당시 PENDING(Priority), 3시간 할당이었다. 기존 pp_ job 8개도 모두 PENDING이었다.
- `queue_s/todo_a00_runtime.jsonl`에 `i.ref.ki.1e-6.g64`와 `i.ref.ki.1e-6.g256` 두 항목을 추가했다. bin/pp_engine_i, target 6M, truncEvery 10, 기본 gmax=8/chunk/window, trace 1, vram fraction 0.9다. 예상 600/240초, 각 1,200초 제한이며 기존 목록 앞쪽에서 실행한다. 새 Slurm job/loop를 만들지 않았고 실행 중인 바이너리를 빌드하지 않았다. 각 ID의 done/log/lock이 없는지와 기존 실행기의 항목/바이너리 선택을 검증했다. 256 GPU의 다른 기존 할당이 먼저 claim해도 기존 lock으로 중복 실행을 막는다.
- Figure 17은 세 cutoff의 25개 완료 값과 KI 2개 Pending이다. 미측정을 0으로 그리지 않는다. 16 GPU도 `logs/r/eng.*`의 재측정으로 통일해 전체 시간 분석 자료 103개가 모두 pp_engine_i, cap 256이다. 분석기는 두 보충 로그가 생기면 각각 독립적으로 읽을 수 있다. higher-cap/chunk 자료는 기존 별도 분석에 남긴다.
- Evaluation의 일부 눈금이 비었던 원인은 겹침 방지용 생략 코드였고, 괄호는 그 코드의 축약 단계에서 제거되거나 fixed-row 생성기에서 처음부터 빠졌다. 처음에는 모든 눈금을 엇갈려 표시했으나, 사용자가 긴 라벨의 중간 생략을 허용하고 cutoff 축의 여러 줄 배치를 원하지 않는다고 명확히 수정했다. 최종 규칙은 cutoff와 `(항 수)` 또는 `(오차)`를 한 줄에 유지하고, 겹치면 중간 눈금만 생략하며 양 끝은 남기는 것이다. Figure 17은 cutoff 세 개만 한 줄로 표시한다. 그림의 측정점 자체는 생략하지 않는다.
- Figure 9/11은 4+3 배치와 행별 동일 플롯 크기를 유지한다. Figure 16의 항 수 괄호를 복원했다. Figure 15에는 Execution Configuration 축 제목을 넣고, 범례의 규모를 intermediate 합산값(22.5M/1.4B)에서 본문 정의에 맞는 boundary 항 수(2.33M/159M)로 바로잡았다. Accuracy의 괄호는 항 수가 아니라 오차라는 구분을 유지한다.
- 사용자가 기존 Table 4(Requirements of distributed Pauli propagation)의 역할을 지적했다. R1–R6은 후속 Design/Evaluation에서 쓰이지 않고 Background를 반복하며, R6의 작은 문제에서 손실이 없어야 한다는 요구는 결과보다 강했다. 해당 표와 참조를 삭제하고 Background의 locality/adaptation/capacity 관찰을 Design의 세 메커니즘으로 잇는 두 문장으로 대체했다. Evaluation 순서를 요구사항 번호에 맞춰 재편하지 않았다. 이제 Runtime parameters가 Table 4, fixed-row window/traffic이 Table 5다.
- 최종 검증: 컴파일 성공, check_style 0건, 미해결 참조/인용과 Overfull 없음. 23쪽, 본문 8,674단어(Evaluation 2,804단어). Evaluation PDF 64개를 갱신했고 Background와 설계 그림은 모두 기존 SHA-256과 같다. 최종 6쪽 및 12–21쪽을 렌더링해 확인했다. 최종 패널 생성의 겹침/넘침 검사도 0건이다. 미리보기는 `paper/build/cutoff-final-*.png`와 `table-removal-final.png`에 있다. 보충 측정 두 건은 아직 미완료이며 기다리지 않고 현재 원고를 저장한다.
