# Pauli propagation 분할 실행 엔진 — 논문 프레이밍 (작업 문서)

상태: 2026-10-07 기준. 통합 엔진(`pp_engine`)으로 회로 7종을 Perlmutter 1–16 GPU에서 쟀다. 시스템 이름은 미정이다.
이 문서는 논문을 어떤 틀로 쓸지, 어떤 주장을 어떤 데이터로 받칠지, 무엇이 아직 비어 있는지를 정리한다.

---

## 1. 한 문단 요약

Pauli propagation은 100 qubit이 넘는 회로의 기대값을 고전 컴퓨터로 근사하는 방법이다. 정확도는 유지하는 항(term)의 수로 정해지는데, 지금의 GPU 구현(NVIDIA cuPauliProp)은 GPU 한 장에 약 1억 항까지만 담고, GPU를 늘리면 오히려 느려진다. 우리는 그 원인이 "truncation 전에 같은 항을 모두 합쳐야 한다"는 제약을 gate마다 전역 조율로 푸는 데 있음을 측정으로 보인다. 그리고 항 집합을 parity로 나누면 묶음들이 gate 수백 개 동안 서로 완전히 독립이 된다는 성질을 찾아, 이 하나의 성질에서 통신 없는 다중 GPU 실행, GPU 메모리를 넘는 실행, checkpoint를 모두 끌어낸다. 결과는 단일 GPU에서 전부 합친 실행, 그리고 cuPauliProp 자체 분산 모드와 12자리까지 같다. 같은 연산 목록을 16 GPU에서 돌리면 cuPauliProp 분산 모드보다 3–190배 빠르고, cuPauliProp이 메모리 부족으로 멈추는 설정도 끝까지 돈다.

---

## 2. 겨냥하는 문제: state-vector 시뮬레이터와 다르다

ScaleQsim, Atlas, HyQuas, BMQSim은 모두 state-vector 시뮬레이터다. 이 연구는 그 계열의 다음 버전이 아니다. 다루는 대상, 한계를 정하는 자원, 정확성의 뜻이 모두 다르다.

| | State-vector 계열 (ScaleQsim, Atlas, HyQuas, BMQSim) | 이 연구 |
|---|---|---|
| 다루는 대상 | 상태 벡터: amplitude 2^n개 | 관측량(operator): 계수가 붙은 Pauli string의 합 |
| 데이터 크기 | qubit 수로 고정 | gate마다 늘고 줄며, 사용자가 cutoff로 정함 |
| 겨냥하는 회로 | 수십 qubit, 전체 상태를 정확히 | 100 qubit 이상, 기대값만, 근사 허용 |
| 한계를 정하는 것 | 메모리 2^n, global qubit 통신 | 유지할 수 있는 항 수, truncation 전 merge |
| 데이터가 놓인 곳 | index로 미리 앎 | 어떤 항이 생길지 실행해 봐야 앎 |
| 부하 | 균일 | 데이터에 따라 다름 |
| 정확성 기준 | exact 상태와의 fidelity | 전부 합친 단일 장치 실행과 같은 답, 그리고 cutoff 수렴 |
| 확장 축 | qubit 수 | 항 수 (얼마나 낮은 cutoff까지 갈 수 있나) |
| 비교 대상 | qsim, cusvaer, Atlas, HyQuas | cuPauliProp, ORQA, PauliPropagation.jl |

이 차이 때문에 논문의 틀도 달라야 한다.

- **비교 축은 "qubit 수 대 시간"이 아니다.** 127 qubit의 상태 벡터는 amplitude가 2^127개라 state-vector 시뮬레이터의 대상이 아니다. 우리의 축은 "항 수(cutoff) 대 GPU 수"다.
- **state-vector 시뮬레이터는 baseline이 아니다.** 같은 문제를 풀지 않는다. 서론 한 문장과 관련 연구 한 문단에서만 다룬다.
- **"더 빠른 시뮬레이터"로 쓰지 않는다.** ScaleQsim식 구성(분할 설계 → SOTA 대비 속도 → 확장성)을 그대로 쓰면 "방법만 바꾼 또 하나의 시뮬레이터"로 읽힌다. 이 논문은 "측정으로 찾은 원인 → 원리 하나 → 그 원리에서 나오는 기능들"로 쓴다.
- **BMQSim과의 관계:** BMQSim은 손실 압축으로 정확도를 조금 내주고 메모리를 얻는다. Pauli propagation은 truncation이 이미 그 역할을 한다. 우리는 truncation의 뜻은 그대로 두고 메모리 한계만 없앤다.

연구실 계보에서는 ScaleQsim(state vector), SWIFTN(tensor network)에 이은 세 번째 방법(operator 공간)이다.

---

## 3. 한 문장 주장과 핵심 원리

**주장:** Pauli propagation에서 항 집합을, 지금 구간의 gate들이 바꾸지 않는 parity로 나누면 묶음들은 그 구간 동안 항을 주고받지도 합쳐지지도 않는다. 따라서 묶음마다 따로 계산해도 전부 합쳐 계산한 것과 같은 답이 나온다.

### 용어

| 용어 | 뜻 |
|---|---|
| 항(term) | Pauli string 하나와 계수. Pauli string은 bit 벡터 (x \| z)로 저장한다 (127 qubit이면 254 bit) |
| rotation | exp(-iθQ/2). Q와 commute하는 항은 그대로이고, anticommute하는 항 P는 P와 P⊕Q 두 항이 된다 |
| merge (dedup) | 같은 Pauli string을 가진 항들의 계수를 더해 하나로 만드는 일 |
| truncation | 계수 크기가 cutoff 이하인 항을 버리는 일. 정확도와 항 수를 정한다 |
| 행(row) | bit 벡터에 대한 parity 함수 하나. r(P) = (P와 mask의 AND에서 1의 개수) mod 2 |
| 묶음(bucket) | 행 g개의 값이 같은 항들의 모음. 최대 2^g개 |
| 구간(window) | 같은 행을 쓸 수 있는 연속된 gate들. kicked Ising에서는 층 하나(rotation 127개) |
| 재분배(re-key) | 구간이 바뀔 때 새 행으로 항을 다시 나누는 일 |

### 원리

1. rotation의 새 항은 P⊕Q다. 행 r이 r(Q)=0이면 r(P⊕Q)=r(P)다.
2. 구간의 모든 생성자 Q에 대해 r(Q)=0인 행들로 묶음을 정하면, 새 항도 합쳐질 상대도 같은 묶음 안에 있다. 묶음은 그 구간 동안 닫혀 있다.
3. 그래서 묶음 안에서만 merge하고 truncation해도 전체를 merge한 것과 같다.
4. Clifford gate는 bit 벡터의 선형 변환이다. 데이터에 적용하지 않고 뒤따르는 rotation의 생성자를 바꿔 흡수할 수 있다.

### 보장과 한계

원리에서 따라 나오는 성질이다. 1–4는 증명할 수 있고, 5–6은 측정으로 확인했다.

| # | 성질 | 뜻 |
|---|---|---|
| 1 | **존재 보장.** 구간의 생성자들이 만드는 공간의 차원이 2n−g 이하이면 유효한 행 g개가 있다 | 어떤 회로든 rotation 2n−g개까지는 통신 없이 한 구간으로 돈다. 127 qubit, 묶음 256개면 246개 |
| 2 | **길게 잡는 것이 최적.** 구간을 가능한 한 길게 잡으면 구간 수가 최소다 | 차원 조건은 부분 구간에서도 성립하므로, 미리 계획하지 않고 실행하면서 늘려 가도 손해가 없다 |
| 3 | **나눌 수 있는 한계.** 묶음은 생성자 공간의 coset들의 합집합이다 | coset 하나에 항이 몰려 있으면 어떤 행으로도 고르게 나눌 수 없다. 표본으로 미리 알 수 있다 |
| 4 | **재분배는 피할 수 없다.** 회로 전체에서 유효한 행은 모든 항에서 값이 같다 | 그런 행은 나누는 데 쓸 수 없고, 대신 검산에 쓴다 |
| 5 | **묶음은 축소판이다.** 한 구간에서 모든 묶음은 같은 연산을 서로 독립으로 받는다 | 묶음 하나를 먼저 돌려 보면 전체의 항 수 변화를 알 수 있다 (pilot) |
| 6 | **쌍은 같은 묶음에 있다.** rotation은 {P, P⊕Q} 쌍 위의 2×2 회전이고, 쌍의 두 항은 같은 묶음에 있다 | 정렬·dedup·출력 버퍼 없이 제자리에서 갱신할 수 있다. 아직 만들지 않았다 |

---

## 4. 기여

1. **해부:** cuPauliProp v0.5.0 분산 모드를 gate 단위로 계측해, GPU를 늘릴수록 느려지는 원인과 GPU당 용량 한계를 수치로 보인다.
2. **원리:** 구간 동안 보존되는 parity로 나눈 묶음은 독립이라는 성질, 재분배가 필요하다는 한계, Clifford를 데이터에 적용하지 않는 방법을 제시한다.
3. **시스템:** 이 원리에서 네 가지를 끌어낸다. gate 중 통신이 없는 다중 GPU 실행, 묶음을 task로 삼는 동적 스케줄링, GPU 메모리를 넘는 실행(넘치는 묶음은 그 자리에서 분할), 층 경계 checkpoint다.
4. **평가:** 회로 7종에서 단일 GPU, cuPauliProp 분산 모드와 같은 답을 낸다. 16 GPU에서 cuPauliProp 분산 모드보다 3–190배 빠르다. cuPauliProp이 16장을 모두 써야 도는 문제를 1장으로 돌리고, 16장으로도 못 도는 문제를 끝낸다. 30 qubit 격자에서는 exact 값과 대조해 cutoff를 낮출수록 오차가 줄어듦을 보인다.

---

## 5. 스킴 목록

스킴은 세 층으로 나뉜다. 맨 아래가 "어떻게 나누나", 그 위가 "나눈 묶음을 어디에 두고 누가 처리하나", 맨 위가 "믿을 수 있게 하는 장치"다. 모든 스킴은 통합 엔진 `pp_engine` 하나에 들어 있다 (`src/engine/`). 앞서 만든 프로토타입(`src/proto/lo_*`)은 초기 측정 기록으로 남긴다.

### 가. 나누는 규칙 (모든 스킴의 바탕)

| 스킴 | 하는 일 | 푸는 문제 | 상태 |
|---|---|---|---|
| Parity 분할 | 구간의 gate가 바꾸지 않는 parity로 묶음을 정한다 | gate마다 전역 dedup | 실측 |
| 실행 중 행 선택 | 지금 항 분포의 표본으로 균형 좋은 parity를 고른다 | 묶음 크기 치우침 | 실측 |
| 재분배 (re-key) | 구간이 바뀔 때 한 번 새 parity로 다시 나눈다 | parity는 영원히 유효할 수 없음 | 실측 |
| 재분배 시점 | 구간을 truncation 직후에서 끝낸다 (`--snap`, 기본). 재분배는 모든 항을 옮기므로 항이 적을 때 한다 | 재분배 때 옮기는 양 | 실측 (XXZ에서 옮기는 항 2.8배 감소) |
| Pilot 묶음 | 묶음 하나의 1/16을 구간 끝까지 먼저 돌려 항 수를 기록하고, 항이 가장 적은 truncation 직후에서 구간을 끝낸다 (`--pilot 1`, 기본은 끔) | 위와 같음 | 실측. 작은 설정에서는 17–26% 빠르지만 16 GPU 큰 설정에서는 2–10% 느려 기본에서 뺐다 |
| Clifford frame | Clifford gate와 Clifford 각도 rotation을 데이터에 적용하지 않고 뒤따르는 rotation의 생성자(부호 포함)에 흡수한다. noise 앞과 맨 끝에서만 데이터에 적용 | gate의 절반이 데이터를 훑기만 함 | 실측 (임의의 H, S, CZ, Pauli gate, π/2 rotation) |
| 구간 자동 결정 | rotation 생성자가 만드는 공간을 GF(2) 소거로 추적해, 남는 parity 공간이 n차원 이상인 동안 구간을 늘린다. 구간 안의 Clifford는 bit 변환으로 따라간다 | 회로마다 손으로 규칙을 넣어야 함 | 실측 (회로 7종에서 구간 6–40개가 자동으로 잡힘) |

### 나. 실행 방식 (묶음을 어디에 두고 누가 처리하나)

| 스킴 | 하는 일 | 푸는 문제 | 상태 |
|---|---|---|---|
| 분산 실행 (MPI) | GPU마다 묶음 하나를 들고 gate를 적용. gate 중 통신 없음, 재분배만 MPI | 다중 GPU가 단일보다 느림 | 실측, 1–16 GPU (`lo_mpi`) |
| Out-of-core | 묶음을 host memory에 두고 GPU에 하나씩 올려 한 층을 처리 | GPU당 약 1억 항 한계 | 실측, GPU 1장 (`lo_ooc`) |
| 묶음 task 스케줄링 | 여러 GPU가 같은 host 저장소에서 묶음을 꺼내 처리. GPU끼리 주고받지 않음 | 재분배의 MPI 비용, 치우침 | 실측, 한 노드 4 GPU (`lo_pool`) |
| 넘치면 분할 | 층 중간에 GPU보다 커진 묶음을 같은 계열 parity로 둘로 쪼갬 | 층 안에서 항이 약 10배로 불어남 | 실측 |
| 고정된 host chunk 풀 | 묶음을 고정 크기 chunk의 목록으로 두고, chunk는 한 번 고정(pin)해 계속 재사용. GPU 복사와 노드 간 전송이 같은 메모리를 씀 | host 복사가 느림 | 실측 (gpu→host 1.1 GB/s → 22 GB/s) |
| 작게 시작 | 항이 적은 동안은 묶음 하나, GPU 한 장만 씀 | 작은 문제에서 분산이 손해 | 실측 |

통합 엔진은 host 저장소 방식 하나로 합쳤다. 노드마다 프로세스 하나, GPU마다 worker 하나가 있고, 묶음 d는 노드 (d mod 노드 수)가 맡는다. 구간이 끝나면 각 노드가 다른 노드 몫의 조각을 host memory에서 묶음 통째로 보낸다. 구간이 도는 동안에는 MPI 호출이 없고 GPU끼리는 주고받지 않는다. 그래서 out-of-core는 여러 노드에서도 그대로 동작하고, 담을 수 있는 항 수는 GPU memory가 아니라 모든 노드의 host memory가 정한다.

### 다. 믿을 수 있게 하는 장치

| 스킴 | 하는 일 | 상태 |
|---|---|---|
| 구간 경계 checkpoint | 묶음 집합과 그 구간의 parity를 노드별 파일로 쓰고, 같은 노드 수 또는 한 노드로 재개 | 실측 (한 노드, 그리고 4 노드에서 쓰고 4 노드에서 재개) |
| 보존 parity 검산 | 회로 전체에서 값이 변하지 않는 parity로 묶음을 올릴 때마다 모든 항을 검사 | 실측 (kicked Ising 1,890만 항 검사, 위반 0) |
| 기준 모드 | 같은 연산 목록을 묶음 하나·GPU 하나로(`--mode ref`), 또는 cuPauliProp 분산 모드로(`--mode cupp`) 돌려 답과 시간을 대조 | 실측 |

### 라. 아직 만들지 않은 것

| 스킴 | 하는 일 |
|---|---|
| 제자리 rotation kernel | 성질 6을 써서 출력 버퍼와 workspace 없이 갱신 (메모리 약 7배는 추정치) |
| GPU에 머무르기 | 묶음이 GPU에 다 들어가면 구간 사이에 host로 내리지 않는다 |
| 더 빠른 노드 간 교환 | 지금은 노드당 thread 하나가 보낸다. 구간이 도는 동안 미리 보내는 방식(`--stream 1`)은 만들었지만 이득이 없었다 (묶음 출력이 구간 끝에 몰린다) |

### 마. 데이터로 기각한 것

합치지 않는 독립 chunk, 재분배 때 덜 움직이는 parity 고르기, 아무 일도 안 하는 gate 건너뛰기, 구간을 coset 기준으로 최대한 늘리기(`--window coset`). 마지막 것은 kicked Ising에서 구간을 20개에서 17개로 줄였지만 구간 끝이 항 수가 가장 많은 지점에 걸려 2.6배 느렸다. 재분배 횟수보다 재분배 시점이 중요하다는 것이 여기서 나왔고, pilot 묶음은 그 결과다. 수치는 7절 표에 있다.

---

## 6. 논문 구조

| 절 | 내용 | 그림·표 |
|---|---|---|
| 1. 서론 | 겨냥하는 문제(항 수가 정확도), 지금의 문제, 원리 한 줄, 결과 | 그림 1: (a) GPU를 늘리면 느려짐 (b) cutoff에 따라 답이 달라짐 (c) cuPauliProp이 16장을 다 써야 도는 문제를 1장으로, 16장으로도 못 도는 문제를 16장으로 |
| 2. 배경 | Pauli propagation 한 쪽. rotation, merge, truncation | 도식 1장 |
| 3. 지금 방식의 해부 | cuPauliProp 분산 모드 측정. 관찰 5개 | gate당 시간 대 항 수, 시간 분해, 용량 표, merge 없이 자르면 틀림 |
| 4. 원리 | 이 문서 3절의 원리를 정리와 증명 스케치로 | 묶음이 닫혀 있음을 보이는 도식 |
| 5. 설계 | 원리에서 나오는 네 기능과 검산 | 구조도, 층 하나의 흐름(적재 → gate 127개 → 분할 → 저장) |
| 6. 평가 | 질문별로 구성 (아래) | |
| 7. 관련 연구 | Pauli propagation 계열, state-vector 계열, DB의 분할 실행 | 비교 표 |
| 8. 한계 | 이 문서 10절의 내용 | |

### 평가 질문

| 질문 | 지금 가진 답 |
|---|---|
| Q1. 답이 같은가 | 회로 7종 모두 단일 GPU, cuPauliProp 분산 모드와 12자리 일치 |
| Q2. 어디까지 풀 수 있나 (cutoff 대 GPU 수) | kicked Ising cutoff 1e-5: cuPauliProp은 4·8 GPU에서 실패, 16 GPU에서만 완료. 엔진은 1 GPU부터. cutoff 3e-6: cuPauliProp 16 GPU 실패, 엔진 133초 |
| Q3. 같이 풀 수 있는 문제에서 얼마나 빠른가 | 16 GPU: 중간 크기 13–187배, 큰 크기 3.2–4.7배 |
| Q4. GPU를 늘리면 줄어드는가 | kicked Ising cutoff 1e-5: 4/8/16 GPU = 46.8/25.7/15.9초 |
| Q5. 비용은 어디에 드는가 | 16 GPU에서 gate 적용이 전체의 약 55%, 노드 간 교환이 13–18%, 나머지는 host 복사·대기·계획 |
| Q6. 다른 회로에서도 되는가 | 회로 7종 모두 16 GPU에서 실측 |
| Q7. 정확도가 실제로 좋아지는가 | 30 qubit 격자 4 step: cutoff 1e-3에서 1e-7까지 오차 0.019 → 0.00007. kicked Ising(θ=π/4)은 아직 수렴하지 않음 |

---

## 7. 주장별 근거 현황

실험 회로는 127-qubit kicked Ising(IBM Eagle heavy-hex, θ=π/4, 20 step, 관측량 Z_62)이고 GPU는 Perlmutter A100-40GB다.

| 설정 이름 | cutoff | weight truncation | 최대 항 수 |
|---|---|---|---|
| 예제 설정 (NVIDIA 예제 기본값) | 1e-4 | 8 | 12만 |
| 2,250만 항 설정 | 1e-4 | 없음 | 2,250만 |
| 14억 항 설정 | 1e-5 | 없음 | 약 14억 (층 경계에서는 1.6억) |
| cutoff 3e-6 설정 | 3e-6 | 없음 | 층 경계에서 15억 (층 안 최대는 재지 않음) |

### 해부 (3절)

| 주장 | 수치 | 상태 |
|---|---|---|
| GPU를 늘리면 느려진다 | 2,250만 항 설정: 1 GPU 2.9–3.2초, 4/8/16 GPU 33–48/65/71초 | 실측 |
| 작은 문제일수록 심하다 | 예제 설정: 1 GPU 0.42초, 4 GPU 32–47초 | 실측 |
| gate마다 고정 비용이 든다 | 4/8/16 GPU에서 gate당 5.9/22.5/29.8 ms. 단일 GPU는 0.05–0.8 ms | 실측 |
| 항이 움직이지 않아도 비용을 낸다 | rank마다 gate당 allgather 22회. gate의 57%는 교환이 없음 | 실측 |
| GPU는 대부분 쉰다 | 사용률 9%. rank 0 CPU 시간의 47%가 MPI polling, 20%가 GPU 메모리 할당·해제, 1.5%가 kernel | 실측 |
| 부하 불균형이 원인이 아니다 | 최대/평균 1.001, 느린 rank 대기 0.2% | 실측 |
| 손으로 조정해도 부족하다 | 10 gate마다 전역 dedup: 4 GPU 9.3초, 16 GPU 14.6초 | 실측 |
| GPU당 용량은 약 1억 항이다 | 실패 지점: 1 GPU 9,600만, 4/8/16 GPU에서 rank당 9,580만/1억 370만/8,790만 | 실측 |
| truncation은 합친 계수를 봐야 한다 | 합치지 않고 자르면 0.194–0.283 (정답 0.1531) | 실측 |
| 정확도는 항 수가 정한다 | cutoff 1e-4, 1e-5, 3e-6에서 0.1531, 0.2286, 0.2892 | 실측 |

### 원리와 설계 (4–5절)

| 주장 | 수치 | 상태 |
|---|---|---|
| 묶음별 계산이 전체 계산과 같다 | 기대값 0.153108187591이 아래 모든 방식에서 일치 | 실측 |
| | MPI 1/4/8/16 GPU, Clifford frame 유무 | 실측 |
| | GPU 메모리의 2%만 쓰는 out-of-core (분할 61회) | 실측 |
| | checkpoint 후 다른 묶음 구성으로 재개 | 실측 |
| | 한 노드 1/2/4 GPU task 방식 | 실측 |
| 큰 문제에서도 같다 | 14억 항 설정: 0.228607441423이 16 GPU(MPI) 두 방식, 1/2/4 GPU(out-of-core)에서 일치 | 실측 (외부 기준값 없음) |
| 다른 parameter에서도 같다 | θ=0.15π, cutoff 1e-5: 0.771759253455 일치 | 모사 (한 GPU 안 가상 rank) |
| 일반 각도 ZZ에서도 된다 | X-bit 행과 Z-bit 행을 번갈아 써서 step당 재분배 2회. 두 설정에서 12자리 일치 | 모사 |
| 균형은 실행 중 고른 행으로 맞춘다 | 16분할: 고정 bit 1.4–4.9배, 고른 parity 1.00–1.01배 | 실측 |
| Clifford를 데이터에 적용하지 않아도 된다 | gate 5,420 → 2,540개, 단일 GPU 2.92 → 1.64초, 답 동일 | 실측 |
| 층 경계가 checkpoint에 알맞다 | 층 안 최대 항 수가 경계의 약 10배. 119만 항 checkpoint가 48 MB | 실측 |
| 보존 parity로 검산할 수 있다 | heavy-hex에서 19개. 층 경계 스냅샷 4개에서 위반 0건 | 실측 |
| 재분배는 피할 수 없다 | 영원히 유효한 행은 상수 (증명). 덜 움직이는 행 탐색도 60–91% 이동 | 증명 + 실측 |
| 통신 없는 독립 chunk는 안 된다 | 4개 0.1675, 16개 0.189 (정답 0.1531) | 실측 (기각) |

### 성능과 용량, 초기 프로토타입 (6절)

아래는 통합 엔진 이전의 프로토타입으로 잰 것이다. 최종 수치는 뒤의 "통합 엔진 대 cuPauliProp 분산 모드" 표를 쓴다.

| 주장 | 수치 | 상태 |
|---|---|---|
| 같은 회로에서 cuPauliProp 분산보다 빠르다 | Clifford frame을 양쪽에 적용: 4 GPU 2.29초 대 17.1초, 16 GPU 2.39초 대 41.2초 | 실측 |
| 원래 회로 기준 | 4 GPU 2.43초 대 32.9초, 16 GPU 3.29초 대 71.2초 | 실측 |
| 전원 통신이 줄어든다 | rank당 collective 51회 대 allgather 119,785회. gate 중 MPI 호출 0회 | 실측 |
| 이동 데이터가 줄어든다 | 4 GPU에서 0.57 GB 대 20.3 GB | 실측 |
| cuPauliProp이 16장을 다 써야 도는 문제를 1장으로 푼다 | 14억 항 설정: 1 GPU 212초, host memory 7 GiB. cuPauliProp은 4·8 GPU에서 메모리 부족, 16 GPU에서 51초 | 실측 |
| GPU를 늘리면 줄어든다 | 같은 설정 1/2/4 GPU = 212/116/64초 (한 노드, MPI 없음). 16 GPU(MPI, GPU 안에서) 11.8초 | 실측 |
| 더 낮은 cutoff도 푼다 | cutoff 3e-6 설정(층 경계 15억 항): 한 노드 4 GPU 676초, 4 노드 16 GPU 221초. 기대값은 두 실행에서 0.289217470624로 일치 | 실측 |
| 64 GPU 이상 | | 없음 |

### exact 값과의 대조 (30 qubit 격자)

5×6 정사각 격자 TFIM(θ=0.2π, ZZ −0.3π, 가운데 qubit의 Z). exact 값은 cuStateVec state vector로 구했다 (`analysis/exact_sv.py`, 3×3과 4×3에서 truncation 없는 Pauli propagation과 12자리 일치).

| 깊이 | exact | cutoff 1e-3 | 1e-4 | 1e-5 | 1e-6 | 1e-7 |
|---|---|---|---|---|---|---|
| 4 step | 0.802867 | 0.78343 (오차 0.019) | 0.79018 (0.013) | 0.79905 (0.0038) | 0.80155 (0.0013) | 0.80280 (0.00007) |
| 6 step | 0.790133 | 0.72858 (0.062) | 0.74777 (0.042) | 0.78088 (0.0093) | 0.78341 (0.0067) | — |
| 8 step | 0.698035 | 0.70650 (0.008) | 0.82231 (0.124) | — | — | — |

- 4 step에서는 cutoff를 10배 낮출 때마다 오차가 줄어 1e-7에서 0.00007이 된다. 이때 구간 경계의 항이 82억 개다.
- 6 step도 줄어들지만 고르지 않다 (3e-6에서 0.0115로 1e-5보다 조금 나쁘다).
- 8 step은 1e-4까지에서는 오차가 단조롭게 줄지 않는다. 이 깊이에서는 더 낮은 cutoff가 필요하다.
- 30 qubit은 state vector가 17초에 끝나는 크기다. 이 표의 목적은 정확도 추세를 참값으로 확인하는 것이지 속도 비교가 아니다.

### 재분배 시점 (`--profile`, `analysis/rekey_timing.py`)

| 회로 | 고정 규칙이 옮긴 항 / 가장 좋은 근처 지점 | 구간 안 최대 항 수 / 경계 항 수 |
|---|---|---|
| kicked Ising, 잡음 있는 kicked Ising, ansatz | 1.07–1.14배 | 2.9–7.3배 |
| TFIM (heavy-hex, 정사각) | 1.28–1.38배 | 2.9–4.0배 |
| XXZ | 3.40배 | 5.5배 |
| QAOA | 4.32배 | 35.9배 |

pilot 묶음을 쓰면 왼쪽 값이 1.00–1.21배(QAOA 1.49배)로 내려간다. 오른쪽 열은 묶음을 하나씩 처리할 때 GPU가 동시에 담지 않아도 되는 양이다.

### 통합 엔진 대 cuPauliProp 분산 모드, 16 GPU

두 쪽에 같은 연산 목록을 준다 (Clifford frame으로 줄인 것. cuPauliProp에 가장 유리한 입력이다). cuPauliProp은 rank당 GPU 하나, 엔진은 노드당 프로세스 하나다. 기대값은 모든 행에서 두 쪽이 같다.

중간 크기 (40 GB 노드, 작업 59475516):

| 회로 | 설정 | 엔진 | cuPauliProp 분산 | 배수 |
|---|---|---|---|---|
| Kicked Ising | cutoff 1e-4 | 2.28초 | 30.3초 | 13배 |
| TFIM heavy-hex | 1e-4 | 2.16초 | 65.7초 | 30배 |
| 2D TFIM 11×11 | 1e-4 | 1.71초 | 42.3초 | 25배 |
| XXZ 6×6 | 1e-3 | 0.18초 | 14.6초 | 81배 |
| QAOA 60 qubit | 1e-4 | 0.47초 | 13.3초 | 28배 |
| Ansatz 60 qubit | 1e-5 | 0.11초 | 19.8초 | 187배 |
| 잡음 있는 kicked Ising | 1e-4 | 2.41초 | 76.8초 | 32배 |

큰 크기 (80 GB 노드를 GPU당 36 GB로 제한, interactive 할당 59475947):

| 회로 | 설정 | 엔진 | cuPauliProp 분산 | 배수 | 기대값 |
|---|---|---|---|---|---|
| Kicked Ising | cutoff 1e-5 (최대 14억 항) | 15.9초 | 51.4초 | 3.2배 | 0.228607441423 |
| TFIM heavy-hex | 1e-5 | 16.8초 | 69.8초 | 4.1배 | 0.451392373925 |
| 잡음 있는 kicked Ising | 1e-5 | 13.3초 | 62.8초 | 4.7배 | 0.173318829841 |
| 2D TFIM 11×11 | 1e-5 | 15.0초 | 64.0초 (40 GB 노드) | 4.3배 | 0.824629850051 |
| Kicked Ising | 3e-6 (구간 경계 15억 항) | 132.7초 | 메모리 부족으로 중단 | — | 0.289217470624 |

GPU 수에 따른 변화 (kicked Ising, cutoff 1e-5, 같은 할당):

| GPU 수 | 엔진 | cuPauliProp 분산 |
|---|---|---|
| 4 | 46.8초 | 중단 (rank당 9,580만 항에서 메모리 부족) |
| 8 | 25.7초 | 중단 (rank당 1억 370만 항) |
| 16 | 15.9초 | 51.4초 |

- 앞의 측정에서 "cuPauliProp은 16 GPU로도 실패"라고 한 것은 원래 회로(gate 5,420개)를 줬을 때다. frame으로 줄인 회로를 주면 16 GPU에서 끝난다. 이 실행 덕분에 0.228607441423이라는 값이 우리 분할과 무관한 방식으로도 확인됐다.
- 엔진은 같은 설정을 GPU 1장으로도 돈다 (앞의 `lo_ooc` 212초).

### 엔진 안에서 고른 설정 (16 GPU, kicked Ising cutoff 1e-5)

| 실험 | 결과 | 결정 |
|---|---|---|
| GPU당 묶음 수 1 / 2 / 4 / 8 | 16.0 / 15.5 / 16.5 / 18.7초 | 2 |
| host memory: 재사용하는 고정 chunk 대 pageable chunk | 복사는 3배 빠르지만 전체 시간은 22.4초 대 21.9초로 같음 | 고정 chunk 유지 (노드 간 전송은 2.8배 빨라짐) |
| pilot 묶음 켬 / 끔 (회로 6종) | 켜면 2–10% 느림 | 끔 |
| 교환을 계산과 겹치기 | 16.2초 대 15.9초 | 끔 |
| 4 노드에서 checkpoint 후 4 노드 재개 | 1.09억 항 저장, 재개 후 기대값 동일 | — |

### 엔진이 다룬 가장 큰 크기

| 실행 | 구간 경계 항 수 | 16 GPU 시간 |
|---|---|---|
| 5×6 격자 6 step, cutoff 1e-6 | 121억 | 341초 |
| 5×6 격자 4 step, cutoff 1e-7 | 82억 | 92초 |
| Kicked Ising, cutoff 3e-6 | 15억 | 133초 |

cuPauliProp은 GPU 한 장에 약 1억 항까지 담는다 (16 GPU면 16억).

---

## 8. 비교 대상과 지표

**비교 대상**

| 대상 | 쓰는 방식 |
|---|---|
| cuPauliProp 단일 GPU | 정답 기준, 단일 장치 속도 |
| cuPauliProp 분산 (기본) | NVIDIA 예제 그대로 |
| cuPauliProp 분산 (조정) | Clifford frame 회로, 10 gate마다 전역 dedup 후 truncation. 가장 강한 baseline |
| ORQA 방식 (고정 분배 함수) | GPU로 옮겨 구현하거나 논문 수치 인용. 아직 없음 |

**지표**

- 답: 기대값이 단일 GPU 실행과 몇 자리까지 같은가.
- 용량: 주어진 GPU 수로 돌릴 수 있는 가장 낮은 cutoff(가장 많은 항).
- 시간: 같은 cutoff에서 걸린 시간, cuPauliProp 대비 배수.
- 구조: gate당 전원 통신 횟수, 이동 데이터 양, GPU 사용률.

### 평가 회로 7종

회로는 일곱 가지로 한다 (2026-10-07 결정). 지금 실측은 1번뿐이다.

| # | 회로 | 크기와 연결 | gate 구성 | 시험하는 것 | 엔진 이름 |
|---|---|---|---|---|---|
| 1 | Kicked Ising (IBM 실험 회로) | 127 qubit, heavy-hex | X rotation + ZZ(π/2, Clifford) | 기준 회로, Clifford frame | `ki`, `ki_lab` |
| 2 | TFIM Trotter, 일반 각도 | 127 qubit, heavy-hex | X rotation + ZZ rotation | Clifford가 없을 때 | `tfim_hh` |
| 3 | 2D TFIM Trotter | 정사각 격자 5×6(30 qubit), 11×11(121 qubit) | 2번과 같음 | 더 촘촘한 연결. 30 qubit은 state vector로 exact 값과 대조 가능 | `tfim_sq` |
| 4 | Heisenberg (XXZ) Trotter | 정사각 격자, Néel 상태에서 시작 | XX, YY, ZZ rotation | X와 Z bit를 함께 뒤집는 gate | `xxz` |
| 5 | QAOA MaxCut | 3-regular 무작위 그래프 | H + ZZ rotation(그래프 edge) + X rotation | 격자가 아닌 연결. 관측량이 edge 합 | `qaoa` |
| 6 | Hardware-efficient ansatz | 선형 배치 | 층마다 모든 qubit에 RY, RZ + CZ 얽힘 | 한 층이 구간 둘. Clifford 얽힘 층 | `hea` |
| 7 | 잡음 있는 kicked Ising | 1번과 같음 | 1번 + qubit마다 depolarizing 잡음 | unitary가 아닌 연산 | `ki_noisy` |

일곱 가지 모두 통합 엔진으로 돌고, 작은 설정에서 단일 GPU 기준(`--mode ref`, 묶음 하나 = cuPauliProp 그대로)과 12자리까지 같다. 엔진 쪽은 묶음을 작게 잡아 분할과 재분배가 실제로 일어나게 했다.

| 회로 | 설정 | 기대값 (기준 = 엔진) | 자동으로 잡힌 구간 | 묶음 실행 횟수 |
|---|---|---|---|---|
| 1 `ki` / `ki_lab` | θ=π/4, 20 step, cutoff 1e-4 | 0.153108187591 | 20 | 94 |
| 2 `tfim_hh` | θ=0.2π, ZZ −0.3π, 20 step, cutoff 1e-4 | 0.324547666766 | 40 | 444 |
| 3 `tfim_sq` 5×6 | 8 step, cutoff 1e-4 | 0.822306395555 | 16 | 279 |
| 3 `tfim_sq` 11×11 | 10 step, cutoff 1e-4 | 0.798821812544 | 20 | 291 |
| 4 `xxz` 6×6 | 4 step, cutoff 1e-3 | −0.000927434911496 | 14 | 80 |
| 5 `qaoa` 60 qubit | p=3, cutoff 1e-4 | 47.4783067466 (edge 90개의 합) | 6 | 69 |
| 6 `hea` 60 qubit | 8층, cutoff 1e-5 | 0.198250860182 | 16 | 59 |
| 7 `ki_noisy` | 잡음 0.005, cutoff 1e-4 | 0.156114065827 | 20 | 116 |

- 고른 기준: Pauli propagation 문헌에서 쓰는 종류(TFIM, Heisenberg, QAOA, hardware-efficient ansatz, 잡음 회로)이고, gate 구성·연결·qubit 수·관측량이 서로 다르다.
- 3번의 30 qubit 격자는 "참값" 문제를 푼다. state vector로 exact 값을 구해 cutoff에 따른 오차를 직접 보인다.
- 회로마다 규칙을 손으로 넣지 않는다. 엔진이 rotation 목록에서 구간과 parity를 실행 중에 정한다.
- 위 설정은 답 대조용으로 작게 잡은 것이다. 논문용 크기(각도, 깊이, 관측량, cutoff 범위)는 아직 정하지 않았다. 4번은 cutoff 1e-4에서 단일 GPU에 들어가지 않아 기준이 없고, 6번은 항 수가 적어 분할이 적게 일어났다.

---

## 9. 관련 연구에서의 자리

| 계열 | 대표 | 우리와의 관계 |
|---|---|---|
| Pauli propagation, GPU | cuPauliProp (NVIDIA) | 직접 비교 대상. gate마다 전역 dedup과 재분배. 내부 비공개, 논문 없음 |
| Pauli propagation, CPU 분산 | ORQA (Fugaku, 1조 항) | 고정 분배 함수라 가지치는 gate마다 통신. GPU 없음 |
| Pauli propagation, 알고리즘 | PauliPropagation.jl, sparse Pauli dynamics | 단일 노드. 메모리가 한계라고 스스로 밝힘 |
| State vector | ScaleQsim, Atlas, HyQuas, BMQSim, cusvaer | 다른 문제. global/local qubit 발상은 닮았지만 여기서는 부하가 데이터에 따라 달라 실행 중에 행을 골라야 함 |
| Tensor network | SWIFTN, cuTensorNet | 같은 큰 회로를 겨냥하는 다른 근사법. 비교하지 않음 |
| DB의 분할 실행 | Grace hash join, 분할 집계, buffer pool, checkpoint | 묶음 처리 방식의 원형. 새로운 점은 gate가 계속 key를 바꾸는데도 분할이 유지되는 key를 찾은 것 |

---

## 10. 주장하지 않을 것, 약한 점

**주장하지 않는다**

- 한 GPU에 들어가는 문제에서 단일 GPU보다 빠르다는 것. 2,250만 항 설정은 4 GPU 1.67초, 단일 GPU 1.64초로 같다.
- quantum advantage 실험의 반박. 14억 항 설정의 값은 내부 실행끼리만 맞춰 봤고 외부 기준값이 없다.
- 이 회로(θ=π/4, 20 step)의 수렴한 답. cutoff를 낮출수록 0.1531, 0.2286, 0.2892로 계속 변한다. 우리가 보이는 것은 "같은 cutoff에서 같은 답을 더 적은 자원으로"이지 "참값"이 아니다.
- cuPauliProp이 원리상 느리다는 것. 느림의 일부는 구현(gate마다 메모리 할당)이라 고쳐질 수 있다. 주장은 구조적 차이(gate 중 통신 0, 전원 통신 횟수, 용량)에 둔다.

**약한 점과 지금 상태**

| 약한 점 | 상태 |
|---|---|
| 여러 노드에 걸친 저장소가 없었다 | 해결. 4 노드 16 GPU까지 실측. 64 GPU 이상은 아직 |
| 일반 회로에서 구간을 정하지 못했다 | 해결. 회로 7종에서 자동으로 잡힌다 |
| Clifford frame이 kicked Ising 전용이었다 | 해결. 원래 회로를 넣으면 엔진이 frame으로 줄인다 |
| MPI 재분배가 느렸다 | 해결. 0.55 GB 교환이 1.2–1.4초에서 0.06–0.09초로 |
| out-of-core 비용 | 대부분 해결. GPU 안에서만 도는 프로토타입 11.8초, 엔진 15.9초 (이전 23.3초). 남은 차이는 노드 간 교환이다 |
| 노드 간 교환 속도 | 남음. 전체의 13–18%. 80 GB 노드에서 노드당 약 5 GB/s, 40 GB 노드에서 9–14 GB/s. 노드당 thread 하나가 보내는 것이 한계로 보인다 |
| 회로가 하나였다 | 해결. 7종 모두 16 GPU에서 실측, cuPauliProp 분산 모드와 대조 |
| 관측량이 Z 하나다 | 일부 해결. QAOA는 edge 합. 나머지는 Z 하나 |
| 참값과의 비교가 없었다 | 해결. 30 qubit 격자에서 exact 값과 대조 |
| baseline이 하나다 | 남음. ORQA 방식 비교가 없다 |
| 제자리 rotation kernel | 남음. 만들지 않았다. GPU당 약 1억 항 한계는 kernel이 정한다 |
| 반복 측정이 없다 | 뒤로 미룸 (2026-10-07 결정). 40 GB 노드와 80 GB 노드(36 GB로 제한) 결과가 섞여 있다 |

---

## 11. 논문까지 남은 실험

| 순서 | 실험 | 받치는 주장 |
|---|---|---|
| 1 | 64–1,024 GPU | 규모 |
| 2 | 노드 간 교환을 여러 thread로 | 남은 비용 |
| 3 | 제자리 rotation kernel | GPU당 항 수 (약 7배는 추정치) |
| 4 | 회로 7종의 논문용 크기와 관측량 확정 (IBM 실험의 weight-10/17, 자화) | 일반성 |
| 5 | ORQA 방식 baseline | 비교 |
| 6 | kicked Ising의 수렴 곡선 (cutoff 1e-6 이하), 수렴이 쉬운 θ | 정확도 |
| 7 | 3–5회 반복, 노드 종류 통일 | 신뢰도 (뒤로 미룸) |

---

## 12. 제목, 이름, 학회

- 제목 후보
  - Terms, Not Qubits: Partitioned Pauli Propagation Beyond GPU Memory
  - Conserved Parities Make Pauli Propagation Partitionable
- 이름 후보: **Coset** (묶음이 수학적으로 coset이다), **Sector** (물리에서 보존량으로 나뉜 구역).
- 학회: SC'27이 1순위다 (보통 4월 초 마감, 공지 확인 필요). 규모 실험과 수렴 곡선이 필요하다. 해부 비중을 키우면 SIGMETRICS도 맞는다.

---

## 13. 코드와 결과 위치

```
build.sh                  전부 빌드 (PrgEnv-gnu, cray-mpich, cudatoolkit)
env/                      cuQuantum 26.09 (cuPauliProp 0.5.0)
jobs/env.sh               실행 환경
src/plugin/prof_mpi.c     cuPauliProp MPI plugin 계측판
src/driver/ki_prof.cpp    cuPauliProp gate별 계측 (단일·분산, --frame)
src/driver/ki_split.cpp   단일 GPU 단계 분해, 스냅샷 dump
src/engine/pp_engine.cpp  통합 엔진 (구간 자동 결정, 묶음 저장소, 여러 노드, checkpoint, 기준 모드)
src/engine/circuit.hpp    회로 7종, Clifford frame, 연산의 bit 변환
src/engine/gf2.hpp        GF(2) 소거, parity 공간
src/proto/shuffle.cu      parity 계산·분할 kernel
src/proto/lo_mpi.cpp      rank당 GPU 하나, MPI 재분배 (--frame)
src/proto/lo_virtual.cpp  한 GPU 안 가상 rank 모사
src/proto/lo_ooc.cpp      GPU 1장 out-of-core, checkpoint
src/proto/lo_pool.cpp     한 노드 여러 GPU, 묶음 task (MPI 없음)
analysis/                 요약·균형·검산 스크립트
```

| 로그 | 내용 |
|---|---|
| logs/pp_prof1-59470347.out | cuPauliProp 분산 첫 측정 (1 노드) |
| logs/pp_prof2-59470887.out | 수동 조정, perf |
| logs/pp_prof4n-59470888.out | 8, 16 GPU |
| logs/pp_lo-59471256.out | MPI 프로토타입 |
| logs/pp_lofr-59472166.out | Clifford frame 적용, 같은 회로의 cuPauliProp |
| logs/pp_ooc-59472272.out | GPU 1장 out-of-core |
| logs/pp_pool-59472385.out | 한 노드 1/2/4 GPU task 방식 |
| logs/pp_engine-59474122.out | 통합 엔진 첫 판, 1/2/4 노드 |
| logs/pp_e2a-59475513.out | kicked Ising 노드 수별, 4 노드 checkpoint와 재개 |
| logs/pp_e2b-59475514.out | 회로 7종 큰 설정, exact 값 대조 |
| logs/pp_e3-59475516.out | cuPauliProp 분산 모드 대 엔진, 회로 7종 |
| logs/inter_run1–4.out | interactive 할당에서 한 설정 실험과 최종 비교 |

gate별 기록은 `results/` 아래 작업 번호별 폴더에 있다.

**저장소에 넣는 것:** 코드(`src/`), 빌드·작업 스크립트(`build.sh`, `jobs/`), 분석 스크립트(`analysis/`), 로그(`logs/`), 이 문서. `env/`, `bin/`, `results/`(원자료 약 480 MB), `third_party/`는 넣지 않는다. 원자료는 Perlmutter의 `/pscratch/sd/s/sgkim/kcj/pauliprop/results`에 있다. `src/plugin/orig_mpi.c`와 `src/driver/ibm_eagle.hpp`는 NVIDIA cuQuantum(BSD-3-Clause)에서 가져온 것이고 원래 저작권 표기를 그대로 둔다.

**통합 엔진 쓰는 법**

```
# 단일 GPU 기준 (묶음 하나, cuPauliProp 그대로)
bin/pp_engine --circuit tfim_sq --lx 11 --ly 11 --steps 10 --coef 1e-4 --mode ref
# 한 노드 4 GPU
bin/pp_engine --circuit tfim_sq --lx 11 --ly 11 --steps 10 --coef 1e-5 --gpus 4 --target 4000000
# 4 노드 (노드당 프로세스 하나)
srun -N 4 -n 4 --ntasks-per-node=1 -c 128 --gpus-per-node=4 bin/pp_engine --circuit ki --coef 1e-5 --gpus 4 --target 6000000
```

`--target`은 구간 시작 때 묶음 하나에 담을 항 수다. 층 안에서 약 10배로 불어나므로 GPU 버퍼 크기의 1/20 정도로 잡는다. 넘치면 엔진이 묶음을 쪼갠다.

**실행할 때 주의**

- 작업은 `sbatch jobs/*.sh`로 낸다 (`-A m5320_g -C gpu -q debug`). debug 큐는 사용자당 동시 2개다.
- `--gpus-per-task=1`로 GPU를 rank별로 격리하면 Cray MPICH의 GPU 전송이 멈춘다. `--gpus-per-node=4`로 모두 보이게 한다.
- thrust 코드는 `-gencode arch=compute_80,code=sm_80`로 빌드한다.
- cuPauliProp은 최근 세 버전 모두 correctness 수정이 있었다. 새 설정은 작은 회로에서 답을 먼저 대조한다.
