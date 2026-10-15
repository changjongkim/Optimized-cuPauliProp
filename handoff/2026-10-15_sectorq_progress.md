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
