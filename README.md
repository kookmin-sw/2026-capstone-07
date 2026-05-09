# QUBO 기반 CNN 채널 프루닝 프로젝트

> **졸업작품 캡스톤 디자인 프로젝트**
> CNN 모델의 채널 프루닝 문제를 QUBO 형태로 정의하고, Simulated Annealing 기반 최적화를 통해 효율적인 채널 선택 방법을 연구합니다.

---

## 1. 프로젝트 소개

딥러닝 모델은 이미지 분류, 객체 인식, 자율주행 인지 시스템 등 다양한 분야에서 높은 성능을 보이고 있습니다. 하지만 CNN 기반 모델은 많은 수의 convolution channel과 parameter를 사용하기 때문에 모델 크기, 연산량, 추론 시간 측면에서 부담이 발생합니다.

본 프로젝트는 이러한 문제를 해결하기 위해 **CNN 채널 프루닝(Channel Pruning)** 문제를 **조합 최적화(Combinatorial Optimization)** 관점에서 접근합니다. 특히, 각 채널을 유지할지 제거할지를 이진 변수로 표현하고, 채널 중요도와 채널 간 상관관계를 기반으로 **QUBO(Quadratic Unconstrained Binary Optimization)** 문제를 구성합니다.

구성된 QUBO 문제는 **Simulated Annealing(SA)** 기반 최적화 기법을 통해 해결하며, 선택된 채널만 남긴 CNN 모델을 재구성한 뒤 fine-tuning을 수행하여 정확도와 경량화 효과를 비교합니다.

---

## 2. 프로젝트 핵심 아이디어

### 기존 프루닝 방식의 한계

일반적인 채널 프루닝은 채널의 weight magnitude, activation 크기, 또는 중요도 점수만을 기준으로 개별 채널을 제거하는 경우가 많습니다. 하지만 CNN 내부의 채널들은 서로 독립적으로 동작하지 않으며, 어떤 채널 조합을 유지하느냐에 따라 최종 성능이 달라질 수 있습니다.

### 제안 방식

본 프로젝트에서는 채널 선택 문제를 다음과 같이 정의합니다.

* 각 채널에 대해 이진 변수 `x_i`를 부여
* `x_i = 1`이면 해당 채널 유지
* `x_i = 0`이면 해당 채널 제거
* 채널별 중요도는 QUBO 행렬의 대각 성분에 반영
* 채널 간 상관관계 또는 중복성은 QUBO 행렬의 비대각 성분에 반영
* 원하는 pruning ratio 또는 유지할 채널 수는 penalty term으로 제어

QUBO 목적함수는 다음과 같은 형태로 표현할 수 있습니다.

```text
minimize    x^T Q x
subject to  x_i ∈ {0, 1}
```

이를 통해 단순히 중요도가 낮은 채널을 제거하는 것이 아니라, **정확도 유지에 유리한 채널 조합을 탐색하는 방식**으로 프루닝을 수행합니다.

---

## 3. 전체 시스템 흐름

```text
1. Pretrained CNN Backbone 준비
        ↓
2. Validation 데이터 입력
        ↓
3. Feature Map 추출
        ↓
4. Weight 기반 채널 중요도 계산
        ↓
5. Feature Map 기반 채널 간 상관관계 계산
        ↓
6. QUBO Matrix 구성
        ↓
7. Simulated Annealing으로 채널 선택
        ↓
8. Pruned Model 재구성
        ↓
9. Fine-tuning 수행
        ↓
10. Accuracy, Latency, Params, FLOPs 비교
```

---

## 4. 주요 기능

### CNN 채널 중요도 계산

학습된 CNN 모델의 convolution layer를 대상으로 각 output channel의 weight magnitude를 계산하여 채널별 중요도를 추정합니다.

### Feature Map 기반 상관관계 분석

Validation 데이터를 모델에 통과시켜 각 채널의 feature map을 추출하고, 채널 간 cosine similarity를 계산하여 중복성이 높은 채널 조합을 파악합니다.

### QUBO Matrix 생성

채널 중요도와 채널 간 상관관계를 결합하여 QUBO 행렬 `Q`를 구성합니다. 이 행렬은 채널 선택 문제를 최적화 문제로 변환하는 핵심 역할을 합니다.

### Simulated Annealing 기반 최적화

생성된 QUBO 문제를 Simulated Annealing 방식으로 풀어, 목표 채널 수 또는 pruning ratio를 만족하는 채널 조합을 탐색합니다.

### 모델 재구성 및 Fine-tuning

선택된 채널만 남기도록 CNN 구조를 재구성하고, fine-tuning을 통해 pruning 이후의 성능 저하를 보완합니다.

---

## 5. 실험 설계

본 프로젝트에서는 다음과 같은 비교 실험을 수행합니다.

| 구분                 | 설명                              | 비교 목적               |
| ------------------ | ------------------------------- | ------------------- |
| Baseline CNN       | 프루닝을 적용하지 않은 원본 CNN 모델          | 기준 정확도 및 추론 시간 측정   |
| Classical Pruning  | weight magnitude 등 기존 기준 기반 프루닝 | 기존 방식과의 성능 비교       |
| QUBO-based Pruning | QUBO 행렬과 SA를 이용한 채널 선택          | 조합 최적화 기반 접근의 효과 분석 |
| Fine-tuned Model   | 프루닝 이후 재학습한 모델                  | 최종 성능 회복 정도 확인      |

### 주요 평가 지표

* **Accuracy**: 프루닝 이후 정확도 유지 여부
* **Latency**: 추론 시간 감소 여부
* **Parameters**: 모델 파라미터 수 감소율
* **FLOPs**: 연산량 감소율
* **Pruning Ratio**: 제거된 채널 비율

---

## 6. 기대 효과

본 프로젝트를 통해 다음과 같은 효과를 기대할 수 있습니다.

1. CNN 모델의 불필요한 채널을 제거하여 모델 경량화 가능
2. 단순 기준 기반 프루닝이 아닌 조합 최적화 기반 채널 선택 가능
3. 정확도와 latency 사이의 trade-off 분석 가능
4. QUBO formulation을 활용한 quantum-inspired optimization 연구 경험 확보
5. 향후 Quantum Annealing 기반 최적화로 확장 가능한 구조 마련

---

## 7. 프로젝트 차별점

본 프로젝트의 핵심 차별점은 CNN 채널 프루닝을 단순한 정렬 문제로 보지 않고, **채널 간 관계를 고려하는 조합 최적화 문제**로 정의한다는 점입니다.

기존 방식이 개별 채널의 중요도만을 기준으로 삼는다면, 본 프로젝트는 다음 요소를 함께 고려합니다.

* 채널 하나하나의 중요도
* 채널 간 중복성
* 유지해야 할 채널 수
* 정확도와 경량화 사이의 균형
* QUBO 기반 최적화 가능성

이를 통해 CNN pruning 문제를 보다 구조적으로 분석하고, quantum-inspired optimization 관점에서 접근합니다.

---

## 8. 소개 영상

<!--
프로젝트 소개 영상은 추후 추가 예정입니다.

예시 형식:

[![프로젝트 소개 영상](https://img.youtube.com/vi/영상ID/0.jpg)](https://www.youtube.com/watch?v=영상ID)

또는 아래와 같이 링크를 추가할 수 있습니다.

- 프로젝트 소개 영상: https://www.youtube.com/watch?v=영상ID
-->

---

## 9. 팀 소개

| 이름   | 역할 | 담당 내용                                 |
| ---- | -- | ------------------------------------- |
| 김선우  | 팀원 | QUBO formulation, 프로젝트 문서화, 발표 자료 구성  |
| 김민우 | 팀원 | CNN 모델 학습 및 pruning 코드 구현             |


<!--
팀원 정보는 실제 팀 구성에 맞게 수정하세요.

예시:

| 이름 | 역할 | GitHub | 담당 |
|---|---|---|---|
| 홍길동 | 팀장 | [@github-id](https://github.com/github-id) | 모델 구현 |
| 김철수 | 팀원 | [@github-id](https://github.com/github-id) | QUBO 설계 |
-->

---

## 10. 사용법

<!--
아래 내용은 코드 제출 및 실행 환경이 확정된 후 수정해서 사용하세요.

### 설치 방법

```bash
git clone https://github.com/kookmin-sw/리포지토리명.git
cd 리포지토리명
pip install -r requirements.txt
```

### 실행 방법

```bash
python train.py
python build_qubo.py
python run_sa.py
python fine_tune.py
```

### 예시 실행 흐름

1. CNN baseline 모델 학습
2. Validation data를 이용한 feature map 추출
3. 채널 중요도 및 상관관계 계산
4. QUBO matrix 생성
5. SA solver 실행
6. Pruned model 생성
7. Fine-tuning 및 성능 평가

### 주요 파일 구조

```text
project-root/
├── README.md
├── index.html 또는 index.md
├── requirements.txt
├── src/
│   ├── train.py
│   ├── feature_extraction.py
│   ├── build_qubo.py
│   ├── sa_solver.py
│   ├── prune_model.py
│   └── fine_tune.py
└── results/
    ├── accuracy.png
    ├── latency.png
    └── pruning_result.csv
```
-->

---

## 11. GitHub Pages

프로젝트 소개 페이지는 GitHub Pages를 통해 확인할 수 있습니다.

```text
https://kookmin-sw.github.io/리포지토리명/
```

<!--
실제 배포 후 아래 링크를 수정하세요.

- 프로젝트 페이지: https://kookmin-sw.github.io/capstone-2026-팀번호/
-->

---

## 12. 기술 스택

| 분야                | 사용 기술                                   |
| ----------------- | --------------------------------------- |
| Deep Learning     | Python, PyTorch                         |
| Model Compression | CNN Channel Pruning                     |
| Optimization      | QUBO, Simulated Annealing               |
| Experiment        | Accuracy, Latency, FLOPs, Parameters 비교 |
| Documentation     | Markdown, GitHub Pages                  |

---

## 13. 향후 계획

* 다양한 pruning ratio에 따른 정확도 변화 분석
* Baseline pruning 방식과 QUBO-based pruning 방식 비교
* Fine-tuning epoch에 따른 성능 회복 정도 분석
* Latency 및 FLOPs 측정 자동화
* Quantum Annealing 기반 solver 적용 가능성 검토
* 실험 결과를 GitHub Pages와 발표 자료에 시각화

---

## 14. 참고 키워드

* CNN Channel Pruning
* Model Compression
* QUBO
* Quadratic Unconstrained Binary Optimization
* Simulated Annealing
* Quantum-inspired Optimization
* Fine-tuning
* Accuracy-Latency Trade-off

---

## 15. 마무리

본 프로젝트는 CNN 모델 경량화 문제를 조합 최적화 관점에서 재해석하고, QUBO 기반 최적화 방식을 적용하여 효율적인 채널 선택 방법을 탐구하는 졸업작품입니다. 단순히 모델을 작게 만드는 것을 넘어, 정확도와 추론 효율성 사이의 균형을 분석하고 향후 quantum optimization으로 확장 가능한 연구 방향을 제시하는 것을 목표로 합니다.
