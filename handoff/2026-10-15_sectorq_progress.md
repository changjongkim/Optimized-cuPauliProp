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
