# Deep Learning Application Term Project

Backorder Prediction, 즉 **백오더 발생 가능성 예측** 프로젝트입니다.

백오더는 고객 주문은 들어왔지만 재고나 공급 문제로 바로 출고하지 못하는 상황을 의미합니다. 이 프로젝트의 목표는 제품별 재고, 수요 예측, 판매량, 공급 지표를 보고 **어떤 제품이 백오더로 이어질 가능성이 큰지 미리 예측**하는 것입니다.

Repository: <https://github.com/namelesspark/deep-learning-term-project.git>

---

## 1. 프로젝트를 한 문장으로 설명하면

**수많은 제품 중에서 백오더 위험이 높은 제품을 먼저 찾아내는 분류 모델을 만드는 프로젝트입니다.**

단순히 "맞았다 / 틀렸다"를 보는 문제가 아닙니다. 실제 상황에서는 전체 제품 중 백오더가 발생하는 제품이 매우 적습니다. 따라서 모델이 모든 제품을 "백오더 아님"이라고 예측해도 accuracy는 높게 보일 수 있습니다. 하지만 그런 모델은 실제 업무 활용성이 낮습니다.

그래서 이 프로젝트에서는 accuracy 대신 다음 질문을 더 중요하게 봅니다.

> 모델이 백오더 위험이 큰 제품을 앞쪽에 잘 배치하는가?

이 질문을 평가하기 위해 `PR_AUC`를 주요 비교 지표로 사용합니다.

---

## 2. 외부 Kaggle 노트북 참고 분석

백오더 예측 데이터의 특성과 기존 접근 방식을 파악하기 위해 Kaggle의 공개 노트북들을 함께 확인했습니다. 참고 목적은 단순히 높은 점수를 가져오는 것이 아니라, **같은 데이터에서 어떤 변수들이 반복적으로 중요하게 등장하는지**, 그리고 **어떤 평가 방식은 조심해야 하는지**를 확인하는 것이었습니다.

| 참고 노트북 | 사용 데이터 | 핵심 내용 | 본 프로젝트에 반영한 판단 |
|---|---|---|---|
| [Product Backorder Analysis](https://www.kaggle.com/code/sandeshak/product-backorder-analysis) | `Kaggle_Training_Dataset_v2.csv`, `Kaggle_Test_Dataset_v2.csv` | RandomForest 기반 feature importance와 ROC curve를 확인했습니다. `national_inv`, `sku`, `sales_3_month`, `sales_1_month`, `in_transit_qty`, `forecast_3_month`, `min_bank`, `lead_time` 등이 중요하게 나타났습니다. | 같은 원본 데이터셋을 사용한 분석으로 볼 수 있습니다. 다만 ROC_AUC 중심이고 `sku`가 중요 변수로 들어갈 수 있어, 성능 수치를 직접 비교하기보다는 중요 변수 후보를 확인하는 용도로 참고했습니다. |
| [Backorder prediction - supply chain](https://www.kaggle.com/code/ananyagiliyal7/backorder-prediction-supply-chain) | `Kaggle_Training_Dataset_v2.csv`, `Kaggle_Test_Dataset_v2.csv` | SMOTE, RandomForest, XGBoost, 선택 변수 15개를 사용했습니다. 중요 변수로 재고, 예측 수요, 과거 판매량, 공급 성과, 리드타임, 안전재고, 로컬 백오더 수량이 반복적으로 등장했습니다. | 같은 원본 데이터셋을 사용했습니다. 다만 PR_AUC가 없고 Recall 중심으로 모델을 판단하므로, 성능 비교보다는 파생변수 설계 근거로 참고했습니다. |
| [Supply Chain Analytics Backorder Prediction](https://www.kaggle.com/code/aditya954/supply-chain-analytics-backorder-prediction) | `product_train.csv` 250,078행 | 같은 백오더 컬럼 구조를 가진 작은 샘플 데이터로 RandomForest를 학습했습니다. `sku`를 제거하고 기본 분류 성능을 확인했습니다. | 컬럼 구조는 유사하지만 전체 원본 데이터셋은 아닙니다. 또한 metric 계산 순서가 부정확한 부분이 있어 성능 비교 근거로는 사용하지 않고, `sku` 제거와 문제 구조 확인 정도만 참고했습니다. |
| [Predict Product Backorders with SMOTE and RF](https://www.kaggle.com/code/haimfeld87/predict-product-backorders-with-smote-and-rf) | `Kaggle_Training_Dataset_v2.csv`, `Kaggle_Test_Dataset_v2.csv` | 결측 행을 제거하고 `lead_time`, `sku`를 제외한 뒤 SMOTE와 RandomForest를 적용했습니다. validation에서는 양호해 보였지만 최종 test에서는 Recall이 크게 낮아졌고, 작성자도 SMOTE가 이 데이터에 적합하지 않다고 정리했습니다. | 같은 원본 데이터셋을 사용했습니다. 이 결과는 본 프로젝트의 SMOTE 실험 해석과 일치합니다. 즉 SMOTE는 특정 기준점의 F1을 양호해 보이게 만들 수 있지만, 전체적인 일반화 성능이나 PR_AUC 관점에서는 주의가 필요합니다. |

외부 노트북들에서 공통적으로 확인된 점은 다음과 같습니다.

1. 이 데이터는 백오더 Yes가 1%도 안 되는 매우 불균형한 분류 문제입니다.
2. `national_inv`, `forecast_*`, `sales_*`, `in_transit_qty`, `min_bank`, `lead_time`, `local_bo_qty`, `perf_*` 계열 변수가 반복적으로 중요하게 나타납니다.
3. 단순 accuracy나 ROC_AUC만으로는 실제 백오더 탐지 성능을 설명하기 어렵습니다.
4. SMOTE는 validation에서는 양호해 보일 수 있지만, test나 PR_AUC 기준에서는 성능이 안정적으로 개선된다고 보기 어렵습니다.
5. 따라서 본 프로젝트는 외부 노트북의 중요 변수 힌트를 참고하되, 전처리 누수를 막고 `PR_AUC`를 중심으로 더 엄격하게 모델을 비교했습니다.

추가로 train 데이터에 다시 예측하여 높은 Precision, Recall, F1을 제시한 예시도 확인했지만, 해당 방식은 실제 검증 성능으로 보기 어렵기 때문에 위 비교 표의 주요 참고 근거에서는 제외했습니다.

---

## 3. 현재 폴더 구조

| 경로 | 역할 | 설명 |
|---|---|---|
| `EDA/` | 초기 데이터 탐색 | 결측치, 이상치, 백오더 비율, 변수 분포를 확인한 초기 분석입니다. |
| `notebooks_v1/` | 1차 모델링 실험 | 최초 전처리 기준으로 트리 모델, MLP, FT-Transformer, TabNet, AutoEncoder, SMOTE 등을 비교한 실험입니다. |
| `notebooks_v2/` | 팀원 파생변수 기반 V2 실험 | 팀원들이 제안한 파생변수와 박천상님 참고 노트북의 아이디어를 누수 없이 정리한 뒤, 다시 전처리/학습을 수행한 모델 집합입니다. |
| `notebooks_v3/` | 딥러닝 전용 심화 실험 | 머신러닝 모델은 더 튜닝하지 않고, `processed_v2`를 사용해 딥러닝 모델만 심화 개선한 실험 집합입니다. |
| `team_chunsang_v1/` | 박천상 팀원 참고 모델 | Base MLP와 Skip Connection MLP 실험입니다. 프로젝트의 `notebooks_v1/`, `notebooks_v2/`, `notebooks_v3/` 실험 흐름과 구분하기 위해 별도 폴더로 분리했습니다. |
| `processed/` | 1차 전처리 결과 | `notebooks_v1/02_Preprocess.ipynb`에서 만든 학습용 데이터입니다. GitHub에는 올리지 않습니다. |
| `processed_v2/` | V2 전처리 결과 | `notebooks_v2/01_Preprocess_V2.ipynb`에서 만든 학습용 데이터입니다. V3 딥러닝 실험도 이 데이터를 사용합니다. GitHub에는 올리지 않습니다. |
| `기타 파일/` | 개인 참고 자료 | 가이드라인 PDF, 팀 공유 대본, 참고 문서 등 로컬 보관용 파일입니다. GitHub에는 올리지 않습니다. |

---

## 4. 왜 notebooks_v1, notebooks_v2, notebooks_v3로 나누었나

### team_chunsang_v1: 팀원 MLP 아이디어 확인

`team_chunsang_v1/`은 박천상 팀원이 진행한 MLP 기반 참고 실험입니다.

여기에는 다음과 같은 아이디어가 있었습니다.

- 단순 MLP
- Skip Connection MLP
- 재고/수요 관련 파생변수
- SHAP 기반 변수 해석

다만 이 폴더는 현재 프로젝트 파이프라인과 전처리 기준이 완전히 같지 않습니다. 그래서 그대로 합치기보다는, 유용한 파생변수 아이디어만 가져와서 `notebooks_v2`에서 다시 정리했습니다.

### notebooks_v1: 1차 공통 실험

`notebooks_v1/`은 현재 프로젝트의 1차 공통 실험 폴더입니다. 최초 전처리 기준으로 XGBoost, LightGBM, MLP, FT-Transformer, TabNet, AutoEncoder, SMOTE, AE Hybrid를 비교했습니다.

이 폴더를 `notebooks_v1/`로 부르는 이유는, 이후 `notebooks_v2/`, `notebooks_v3/`와 같은 흐름으로 비교하기 위해서입니다.

### notebooks_v2: 팀원 파생변수와 참고 아이디어를 안전하게 반영

`notebooks_v2/`는 팀원들이 제안한 파생변수와 `team_chunsang_v1/`의 유용한 아이디어를 반영하되, **데이터 누수 없이 다시 학습한 버전**입니다.

여기서 중요한 원칙은 다음과 같습니다.

> train 데이터에서만 기준을 학습하고, validation/test에는 그 기준을 적용만 한다.

예를 들어 평균값, 중앙값, 스케일러, clip 기준, 최대값 같은 것은 전체 데이터를 보고 계산해서는 안 됩니다. test 데이터의 정보를 미리 본 셈이 되기 때문입니다.

### V3: 딥러닝 모델만 심화 개선

`notebooks_v3/`는 딥러닝응용 수업의 성격을 살리기 위해 만든 딥러닝 전용 심화 실험입니다.

V2에서 XGBoost와 LightGBM이 강하게 나왔지만, V3에서는 머신러닝 모델을 더 올리는 것이 목적이 아닙니다. V3의 목적은 다음과 같습니다.

- 기존 MLP보다 성능이 개선된 딥러닝 모델을 만들 수 있는지 확인
- 불균형 데이터에 맞는 딥러닝 기법을 적용
- TabNet처럼 정형 데이터용 딥러닝 모델을 제대로 튜닝
- 최종적으로 딥러닝 모델이 어디까지 따라갈 수 있는지 확인

---

## 5. 현재 데이터의 어려운 점

### 백오더는 매우 적게 발생한다

`processed_v2` 기준 데이터 분포는 다음과 같습니다.

| 데이터 | 전체 행 수 | 백오더 아님 | 백오더 발생 | 백오더 비율 |
|---|---:|---:|---:|---:|
| train | 1,350,288 | 1,341,254 | 9,034 | 약 0.67% |
| validation | 337,572 | 335,313 | 2,259 | 약 0.67% |
| test | 242,075 | 239,387 | 2,688 | 약 1.11% |

이 말은 1,000개 제품 중 백오더가 약 7개 정도밖에 없다는 뜻입니다. 이런 문제에서는 accuracy가 거의 의미가 없습니다.

예를 들어 모든 제품을 "백오더 아님"이라고 예측해도 약 99% 가까운 accuracy가 나올 수 있습니다. 하지만 실제 백오더 제품은 전혀 찾지 못합니다.

그래서 이 프로젝트는 일반적인 정확도보다 `PR_AUC`, `Recall`, `Precision`, `F1`을 중심으로 봅니다.

---

## 6. 주요 평가 지표를 쉽게 설명

### 6.1 Recall

Recall은 **진짜 백오더 제품 중에서 모델이 얼마나 많이 잡아냈는지**를 의미합니다.

예시:

| 실제 백오더 제품 | 모델이 백오더라고 잡은 제품 |
|---:|---:|
| 100개 | 80개 |

이 경우 Recall은 0.80입니다.

즉, Recall이 높다는 것은 **놓치는 백오더가 적다**는 뜻입니다.

하지만 Recall만 높이면 문제가 생길 수 있습니다. 모든 제품을 백오더라고 예측하면 Recall은 1.0이 됩니다. 대신 너무 많은 제품을 위험하다고 경고하게 됩니다.

### 6.2 Precision

Precision은 **모델이 백오더라고 예측한 것 중 실제 백오더가 얼마나 되는지**를 의미합니다.

예시:

| 모델이 백오더라고 예측 | 그중 실제 백오더 |
|---:|---:|
| 1,000개 | 50개 |

이 경우 Precision은 0.05입니다.

즉, 모델이 1,000개를 위험하다고 했는데 실제로는 50개만 백오더였다는 뜻입니다.

현재 프로젝트에서 `threshold = 0.5` 기준 XGBoost의 Precision이 약 0.086이라는 말은 다음처럼 해석할 수 있습니다.

> 모델이 백오더라고 경고한 제품 중 약 8.6%가 실제 백오더였다.

백오더 자체가 매우 희귀하기 때문에 Precision이 낮게 나오는 것은 자연스러운 면이 있습니다. 다만 실제 업무에서는 경고가 너무 많으면 사람이 확인하기 어렵기 때문에 Precision도 중요합니다.

### 6.3 F1

F1은 Precision과 Recall의 균형을 보는 지표입니다.

- Recall만 높이면 너무 많이 경고할 수 있습니다.
- Precision만 높이면 진짜 백오더를 많이 놓칠 수 있습니다.
- F1은 이 둘 사이의 균형을 봅니다.

예를 들어 재고 담당자가 "너무 많이 놓치면 안 되지만, 너무 많은 제품을 경고해도 곤란하다"고 하면 F1이 유용합니다.

다만 F1은 특정 threshold를 정한 뒤 계산됩니다. 따라서 모델 자체의 전체 성능을 비교할 때는 `PR_AUC`를 먼저 보고, 이후 threshold를 조정하면서 F1을 봅니다.

### 6.4 PR_AUC

PR_AUC는 Precision-Recall 곡선 아래 면적입니다.

용어는 어렵지만, 이 프로젝트에서는 다음처럼 이해하면 됩니다.

> 모델이 제품들을 백오더 위험 순서로 얼마나 잘 정렬했는가?

여기서 "줄을 세운다"는 표현이 어렵게 느껴질 수 있습니다. 구체적으로는 모델이 각 제품에 백오더 확률 점수를 붙이고, 그 점수가 높은 제품부터 낮은 제품까지 순서대로 정렬한다는 뜻입니다.

예시:

| 제품 | 모델이 준 백오더 위험 점수 | 실제 백오더 여부 | 위험 순위 |
|---|---:|---|---:|
| A | 0.92 | Yes | 1 |
| B | 0.87 | No | 2 |
| C | 0.81 | Yes | 3 |
| D | 0.30 | No | 4 |
| E | 0.12 | No | 5 |

이 모델은 A와 C처럼 실제 백오더인 제품을 앞쪽에 배치했습니다. 이 경우 백오더 위험 순위를 비교적 잘 잡은 모델입니다.

반대로 다음과 같은 모델은 백오더 위험 순위를 제대로 잡지 못한 모델입니다.

| 제품 | 모델이 준 백오더 위험 점수 | 실제 백오더 여부 | 위험 순위 |
|---|---:|---|---:|
| A | 0.92 | No | 1 |
| B | 0.87 | No | 2 |
| C | 0.81 | No | 3 |
| D | 0.30 | Yes | 4 |
| E | 0.12 | Yes | 5 |

실제 백오더 제품이 뒤쪽에 밀렸기 때문입니다.

PR_AUC가 높다는 것은 threshold를 어디로 잡든, 전반적으로 백오더 위험 제품을 앞쪽에 잘 배치한다는 의미입니다.

### 6.5 ROC_AUC

ROC_AUC도 순위를 보는 지표입니다. 하지만 백오더처럼 Yes가 매우 적은 문제에서는 ROC_AUC가 지나치게 높게 보일 수 있습니다.

예를 들어 백오더가 0.7%밖에 없으면 대부분은 No입니다. 이때 No를 잘 맞히는 것만으로도 ROC_AUC가 높게 보일 수 있습니다.

그래서 이 프로젝트에서는 ROC_AUC를 참고만 하고, 핵심 비교는 PR_AUC로 합니다.

### 6.6 threshold 0.5

모델은 보통 "백오더일 확률" 같은 점수를 냅니다.

예를 들어 어떤 제품의 점수가 0.73이면 모델은 "이 제품은 백오더 위험이 높다"고 본 것입니다.

threshold는 이 점수를 어디서 자를지 정하는 기준입니다.

| 제품 | 점수 | threshold 0.5 기준 예측 |
|---|---:|---|
| A | 0.73 | 백오더 |
| B | 0.48 | 백오더 아님 |
| C | 0.12 | 백오더 아님 |

하지만 0.5가 항상 최적은 아닙니다.

백오더처럼 희귀한 문제에서는 threshold를 낮추면 더 많이 잡지만 경고도 많아지고, threshold를 높이면 경고는 줄지만 놓치는 백오더가 늘어납니다.

그래서 V2와 V3에서는 모델을 고른 뒤 threshold를 따로 조정합니다.

---

## 7. 주요 기법 설명

### 7.1 GBDT, XGBoost, LightGBM

GBDT는 Gradient Boosting Decision Tree의 약자입니다.

쉽게 말하면 작은 의사결정나무를 여러 개 쌓아서 성능을 올리는 방식입니다. 첫 번째 나무가 틀린 부분을 두 번째 나무가 보완하고, 두 번째 나무가 틀린 부분을 세 번째 나무가 다시 보완하는 식입니다.

XGBoost와 LightGBM은 GBDT 계열의 대표 모델입니다.

정형 테이블 데이터에서는 XGBoost와 LightGBM이 매우 강합니다. 이 프로젝트에서도 V2 기준 XGBoost가 가장 높은 PR_AUC를 보였습니다.

### 7.2 MLP

MLP는 가장 기본적인 딥러닝 분류 모델입니다.

이미지처럼 격자 구조가 있거나 문장처럼 순서가 있는 데이터가 아니라, 표 형태의 숫자 데이터를 넣어서 여러 층의 신경망으로 패턴을 학습합니다.

이 프로젝트에서는 딥러닝 모델의 기본 기준선으로 사용했습니다.

### 7.3 TabNet

TabNet은 정형 데이터에 맞게 설계된 딥러닝 모델입니다.

일반 MLP와 달리, 매 단계에서 어떤 변수를 더 볼지 선택하는 구조를 가지고 있습니다. 그래서 변수 중요도도 함께 볼 수 있습니다.

다만 "정형 데이터 특화"라고 해서 항상 XGBoost보다 좋은 것은 아닙니다. 현재 실험에서는 TabNet을 Optuna로 튜닝했지만 MLP_V3보다 낮았습니다. 따라서 TabNet은 최종 성능 모델이라기보다 **비교 실험과 해석 보조 모델**로 보는 것이 적절합니다.

### 7.4 FT-Transformer

FT-Transformer는 정형 데이터에 Transformer 구조를 적용한 모델입니다.

Transformer는 원래 자연어 처리에서 강한 구조입니다. 정형 데이터에서는 각 변수를 토큰처럼 보고 변수 간 관계를 학습하려는 방식입니다.

현재 실험에서는 FT-Transformer가 XGBoost나 MLP보다 낮게 나왔습니다.

### 7.5 AutoEncoder

AutoEncoder는 입력 데이터를 압축했다가 다시 복원하는 모델입니다.

정상 데이터만 잘 복원하도록 학습하면, 비정상 데이터는 복원이 잘 안 되어 재구성 오차가 커질 수 있습니다. 이 프로젝트에서는 백오더를 이상상황처럼 보고 AutoEncoder 기반 이상탐지를 실험했습니다.

하지만 결과적으로 AutoEncoder 단독 모델은 성능이 낮았습니다. 이후 AE 점수를 다른 분류기에 추가하는 하이브리드 실험도 진행했습니다.

### 7.6 SMOTE

SMOTE는 적은 클래스 데이터를 인위적으로 늘리는 방법입니다.

예를 들어 백오더 Yes가 너무 적으면, 기존 Yes 샘플들 사이를 보간해서 가짜 Yes 샘플을 만듭니다.

다만 이 데이터에서는 SMOTE가 항상 좋지는 않았습니다.

XGB_SMOTE는 threshold 0.5 기준 F1은 높았지만, PR_AUC는 기본 XGBoost보다 낮아졌습니다. 즉 특정 기준점에서는 양호해 보일 수 있지만, 전체적인 위험 순위 품질은 떨어졌습니다.

### 7.7 Focal Loss

Focal Loss는 어려운 샘플에 더 집중하게 만드는 손실 함수입니다.

백오더 데이터에서는 대부분이 No입니다. 모델이 쉬운 No만 계속 맞히는 방향으로 학습하면, 희귀한 Yes를 충분히 학습하지 못합니다.

Focal Loss는 모델이 이미 쉽게 맞히는 샘플의 영향은 줄이고, 예측하기 어려운 샘플에 더 집중하게 합니다.

### 7.8 Weighted BCE

BCE는 이진 분류에서 기본적으로 쓰는 손실 함수입니다. 백오더 Yes/No처럼 둘 중 하나를 맞히는 문제에 사용합니다.

Weighted BCE는 Yes 클래스에 더 큰 가중치를 주는 방식입니다.

예를 들어 Yes가 너무 적으면 모델은 No만 맞혀도 손실이 작아질 수 있습니다. 이때 Yes를 틀렸을 때 더 크게 벌점을 주면 모델이 Yes를 더 신경 쓰게 됩니다.

### 7.9 Dropout

Dropout은 학습 중 일부 뉴런을 임시로 꺼서 과적합을 줄이는 방법입니다.

비유하면 매번 같은 학생에게만 문제를 풀게 하지 않고, 여러 학생이 돌아가며 풀게 해서 전체 실력을 키우는 방식입니다.

### 7.10 BatchNorm

BatchNorm은 신경망 내부 값의 분포를 안정화하는 방법입니다.

학습 중 값의 범위가 너무 크게 흔들리면 모델이 불안정해질 수 있습니다. BatchNorm은 각 층의 입력을 어느 정도 정리해서 학습을 더 안정적으로 만듭니다.

### 7.11 Residual MLP

Residual 구조는 입력 정보를 중간층을 건너뛰어 뒤쪽으로 전달하는 구조입니다.

깊은 신경망은 층이 많아질수록 학습이 어려워질 수 있습니다. Residual 구조는 "원래 정보"를 뒤쪽까지 보존하게 해 학습을 안정화합니다.

V3의 MLP_V3는 Residual MLP를 사용했습니다.

### 7.12 Wide & Deep

Wide & Deep은 단순한 정보와 복잡한 정보를 함께 학습하려는 구조입니다.

- Wide: 원본 입력의 단순하고 직접적인 신호를 살림
- Deep: 여러 층을 거치며 복잡한 패턴을 학습

V3에서는 이 구조를 후보로 넣었고, Optuna가 실제로 필요한 설정을 탐색했습니다.

### 7.13 WeightedRandomSampler

WeightedRandomSampler는 학습 배치를 만들 때 희귀한 Yes 샘플이 더 자주 등장하도록 뽑는 방법입니다.

원래 데이터에서 Yes가 0.7%밖에 없으면, 미니배치 대부분이 No로 채워집니다. 그러면 딥러닝 모델은 Yes 사례를 충분히 학습하지 못합니다.

WeightedRandomSampler는 Yes를 더 자주 보여줘서 모델이 백오더 패턴을 학습할 기회를 늘립니다.

### 7.14 Optuna

Optuna는 하이퍼파라미터 자동 탐색 도구입니다.

예를 들어 hidden size, dropout, learning rate, focal loss의 alpha/gamma 같은 값은 사람이 감으로 정하기 어렵습니다. Optuna는 여러 조합을 실험해서 validation 성능이 좋은 조합을 찾습니다.

V3에서는 MLP와 TabNet에 Optuna를 사용했습니다.

### 7.15 Early Stopping

Early Stopping은 validation 성능이 더 이상 좋아지지 않으면 학습을 멈추는 방법입니다.

너무 오래 학습하면 train 데이터에만 맞는 모델이 될 수 있습니다. 이 프로젝트에서는 특히 V3 MLP에서 loss가 아니라 `PR_AUC` 기준으로 early stopping을 적용했습니다.

즉, 손실이 줄어드는지만 본 것이 아니라 **백오더 위험 순위를 잘 세우는지**를 기준으로 모델을 골랐습니다.

### 7.16 SHAP

SHAP은 모델이 왜 그런 예측을 했는지 설명하는 방법입니다.

예를 들어 어떤 제품이 백오더 위험이 높다고 나왔을 때, SHAP을 보면 다음처럼 설명할 수 있습니다.

| 변수 | 예측에 준 영향 |
|---|---|
| 실질 가용재고가 낮음 | 백오더 위험 증가 |
| 3개월 미래 가용재고가 낮음 | 백오더 위험 증가 |
| 미납 주문이 있음 | 백오더 위험 증가 |

즉, SHAP은 성능을 올리는 기법이라기보다 보고서에서 모델 해석을 도와주는 도구입니다.

---

## 8. 파생변수 설명

원본 변수만으로는 백오더 구조를 모델이 바로 이해하기 어렵습니다. 그래서 재고, 수요, 공급 위험을 더 직접적으로 표현하는 파생변수를 만들었습니다.

### 8.1 팀장님 제안 파생변수

`기타 파일/딥러닝응용_1조_전처리_및_파생변수_생성.docx`에 정리된 핵심 파생변수입니다. 현재 전처리에도 반영되어 있습니다.

| 파생변수 | 의미 | 왜 필요한가 |
|---|---|---|
| `available_inventory` | 현재 재고 - 안전재고 | 장부상 재고가 있어도 안전재고를 제외하면 실제 여유가 적을 수 있습니다. |
| `real_available_inventory` | 현재 재고 - 안전재고 - 미납 주문 | 이미 밀린 주문까지 고려한 실제 여유 재고입니다. |
| `future_available_inventory_3m` | 현재 재고 + 운송 중 재고 - 안전재고 - 3개월 예측 수요 | 가까운 미래 수요를 감당할 수 있는지 봅니다. |
| `shortage_ratio_3m` | 3개월 예측 수요 대비 부족 비율 | 제품마다 규모가 다르므로 절대 부족량보다 비율이 더 비교하기 쉽습니다. |
| `no_sales_but_demand_signal` | 판매는 없지만 예측 수요나 미납 주문이 있는지 | 수요가 없는 것이 아니라 재고 부족 때문에 판매가 안 됐을 가능성을 표시합니다. |

### 8.2 기존 전처리에서 추가한 위험 신호

| 파생변수 | 의미 | 왜 필요한가 |
|---|---|---|
| `neg_inv_flag` | 현재 재고가 음수인지 | 재고가 이미 부족한 상태일 수 있습니다. |
| `has_local_bo` | 미납 주문이 있는지 | 이미 공급 지연이 발생했는지 봅니다. |
| `has_past_due` | 지연 수량이 있는지 | 납기 지연 위험을 봅니다. |
| `below_safety_flag` | 현재 재고가 안전재고보다 낮은지 | 안전재고를 깨고 내려간 제품은 위험할 수 있습니다. |
| `sales_acceleration` | 최근 판매 증가 흐름 | 최근 수요가 빠르게 늘면 백오더 위험이 커질 수 있습니다. |
| `forecast_accuracy` | 예측 대비 실제 판매 차이 | 예측이 실제 판매와 얼마나 맞는지 봅니다. |
| `total_risk_score` | 여러 위험 플래그를 합친 점수 | 재고 부족, 미납, 지연, 안전재고 부족을 하나의 위험 신호로 요약합니다. |

### 8.3 V2에서 추가한 박천상님 참고 노트북 기반 안전 피처

박천상님 참고 노트북의 아이디어 중 의미 있는 파생변수를 가져오되, 누수가 생기지 않도록 V2에서 다시 구현했습니다.

| 파생변수 | 의미 | 왜 필요한가 |
|---|---|---|
| `inventory_coverage` | 현재 재고가 3개월 예측 수요를 얼마나 덮는지 | 수요 대비 재고 여유를 봅니다. |
| `stock_shortage_risk` | 예측 수요와 운송 중 재고를 고려한 부족 위험 | 미래 부족 가능성을 직접 표현합니다. |
| `days_of_stock` | 현재 재고가 며칠치 판매량에 해당하는지 | 재고가 며칠 버틸 수 있는지 직관적으로 봅니다. |
| `supply_chain_health` | 운송 중 재고가 부족을 얼마나 보완하는지 | 공급망 보완 상태를 봅니다. |
| `lead_time_demand_ratio` | 리드타임 동안 발생할 수요 규모 | 입고까지 걸리는 시간 동안 수요가 얼마나 쌓이는지 봅니다. |
| `replenishment_urgency` | 보충 필요 긴급도 | 수요 대비 재고와 운송 중 재고가 충분한지 봅니다. |
| `avg_sales` | 여러 기간 판매량의 가중 평균 | 단기/중기 판매 흐름을 하나로 요약합니다. |
| `forecast_volatility` | 3/6/9개월 예측의 변동성 | 예측이 흔들리는 제품은 수요 불확실성이 큽니다. |
| `product_criticality` | 판매 중요도, 리드타임, 예측 변동성을 결합 | 많이 팔리고, 오래 걸리고, 예측이 흔들리는 제품은 관리 우선순위가 높습니다. |

---

## 9. 노트북별 상세 설명

### 9.1 `EDA/`

| 파일 | 설명 |
|---|---|
| `EDA/EDA.ipynb` | 원본 데이터의 결측치, 이상치, 변수 타입, 백오더 분포를 확인한 초기 탐색 노트북입니다. `lead_time` 결측, `-99` 값, 수치형 변수의 치우침, Yes/No 범주형 변수 등을 확인했습니다. |
| `EDA/*.png` | EDA 과정에서 저장된 시각화 이미지입니다. |

### 9.2 `notebooks_v1/` 1차 실험

| 파일 | 설명 |
|---|---|
| `02_Preprocess.ipynb` | 최초 공통 전처리 노트북입니다. 원본 CSV를 불러오고, 불필요한 컬럼 제거, Yes/No 변환, 결측 처리, 파생변수 생성, train/val/test 분할, 중앙값 대치, 표준화를 수행합니다. |
| `03_Baseline_Tree.ipynb` | LightGBM과 XGBoost를 학습합니다. 딥러닝 모델과 비교할 강한 기준선을 만들기 위한 노트북입니다. SHAP 분석도 포함합니다. |
| `04_MLP.ipynb` | 기본 MLP와 불균형 대응 실험입니다. plain BCE, weighted BCE, focal loss, dropout, batchnorm, undersampling 등을 비교합니다. |
| `05_FT_Transformer.ipynb` | 정형 데이터용 Transformer 실험입니다. 변수 간 관계를 attention 구조로 학습하려는 시도입니다. |
| `06_TabNet.ipynb` | TabNet 1차 실험입니다. 정형 데이터 특화 딥러닝 모델과 내장 feature importance를 확인합니다. |
| `07_AutoEncoder.ipynb` | AutoEncoder를 이용해 백오더를 이상상황처럼 탐지해보는 실험입니다. |
| `08_Decision_Analysis.ipynb` | 모델 결과를 통합하고 threshold 0.5의 한계를 분석합니다. F1 최적 threshold, 비용 민감 threshold, 오류 분석을 수행합니다. |
| `09_SMOTE.ipynb` | train 데이터에만 SMOTE를 적용해 XGBoost와 MLP를 비교합니다. |
| `10_AE_Hybrid.ipynb` | AutoEncoder가 만든 재구성오차/이상점수를 원본 피처에 추가해 MLP, XGBoost, FT-Transformer, TabNet을 다시 학습합니다. |
| `results.csv` | 1차 실험 결과 요약 파일입니다. |
| `utils.py` | 공통 평가 함수, 결과 저장 함수 등 노트북에서 사용하는 보조 코드입니다. |

### 9.3 `notebooks_v2/` 팀원 파생변수 기반 재실험

`notebooks_v2/`는 **팀원 파생변수와 박천상님 참고 노트북 아이디어를 반영한 뒤 다시 학습한 모델 집합**입니다.

| 파일 | 설명 |
|---|---|
| `01_Preprocess_V2.ipynb` | V2 전처리 핵심 노트북입니다. 기존 파생변수 12개에 박천상님 참고 노트북 기반 안전 피처 9개를 추가합니다. 결측 대치, clip, log1p, scaling은 train 기준으로만 fit합니다. 결과는 `processed_v2/`에 저장됩니다. |
| `02_Baseline_Tree_V2.ipynb` | V2 피처 기준 XGBoost와 LightGBM을 다시 학습합니다. 기본 설정과 튜닝 설정을 모두 비교합니다. |
| `03_MLP_V2.ipynb` | V2 피처 기준 MLP를 다시 실험합니다. plain BCE, weighted, focal, dropout, batchnorm, SMOTE를 비교합니다. |
| `04_TabNet_V2.ipynb` | V2 피처 기준 TabNet을 다시 학습하고 feature importance를 확인합니다. |
| `05_Ensemble_V2.ipynb` | 단일 모델 확률을 섞는 확률 블렌딩 실험입니다. 여러 모델의 예측 확률 평균 또는 가중 평균을 비교합니다. |
| `06_Threshold_Optimization_V2.ipynb` | V2 최고 모델의 threshold를 조정합니다. F1 최적, F2 최적, 비용 민감 threshold를 비교합니다. |
| `results_v2.csv` | V2 실험 결과 요약 파일입니다. |
| `utils.py` | V2 노트북 공통 함수입니다. |

### 9.4 `notebooks_v3/` 딥러닝 전용 심화 실험

`notebooks_v3/`는 **머신러닝 모델을 제외하고 딥러닝 모델만 심화한 실험 집합**입니다.

중요한 점은 V3가 새로운 전처리를 다시 만든 것이 아니라, V2에서 만든 `processed_v2` 데이터를 사용했다는 것입니다. 즉, V2와 같은 입력 데이터에서 딥러닝 구조와 학습 전략만 개선했습니다.

| 파일 | 설명 |
|---|---|
| `01_MLP_V3_DL_Imbalance.ipynb` | V3의 핵심 딥러닝 실험입니다. Residual MLP, Wide & Deep 후보, WeightedRandomSampler, Focal Loss, Optuna 튜닝, PR_AUC 기준 Early Stopping을 적용합니다. 기존 MLP보다 성능이 상승했습니다. |
| `02_TabNet_V3_Tuning.ipynb` | TabNet을 Optuna로 튜닝한 실험입니다. `n_d`, `n_steps`, `gamma`, `lambda_sparse`, learning rate 등을 탐색하고 feature importance를 확인합니다. |
| `results_v3_dl.csv` | V3 딥러닝 실험 결과 요약 파일입니다. |
| `utils.py` | V3 노트북에서 사용하는 보조 함수입니다. |

### 9.5 `team_chunsang_v1/`

| 파일 | 설명 |
|---|---|
| `team_chunsang_v1/BaseMLP/BaseMLP_결과.ipynb` | 박천상 팀원의 기본 MLP 실험입니다. |
| `team_chunsang_v1/SkipConnectionMLP/SkipConnectionMLP_결과.ipynb` | 박천상 팀원의 Skip Connection MLP 실험입니다. V2에서 일부 아이디어를 안전하게 반영했습니다. |

---

## 10. 현재 모델 결과

### 10.1 1차 실험 결과

Source: `notebooks_v1/results.csv`

모든 threshold 기반 지표는 `threshold = 0.5` 기준입니다.

| Model | PR_AUC | ROC_AUC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.2438 | 0.9618 | 0.8836 | 0.0598 | 0.1120 |
| XGB_AE_hybrid | 0.2348 | 0.9618 | 0.8862 | 0.0611 | 0.1143 |
| LightGBM | 0.2237 | 0.9580 | 0.8907 | 0.0557 | 0.1048 |
| XGB_SMOTE | 0.2127 | 0.9510 | 0.3878 | 0.2272 | 0.2866 |
| MLP_3_focal | 0.1927 | 0.9463 | 0.1983 | 0.3109 | 0.2422 |
| MLP_AE_hybrid | 0.1905 | 0.9432 | 0.1456 | 0.3350 | 0.2030 |
| MLP_SMOTE | 0.1885 | 0.9473 | 0.5091 | 0.1679 | 0.2526 |
| FT_Transformer | 0.1446 | 0.9338 | 0.0713 | 0.2800 | 0.1136 |
| TabNet | 0.0932 | 0.9193 | 0.9172 | 0.0241 | 0.0470 |
| AE_anomaly | 0.0250 | 0.7028 | 0.0000 | 0.0000 | 0.0000 |

1차 실험에서는 XGBoost가 가장 높았고, 딥러닝 모델 중에서는 `MLP_3_focal`이 가장 나았습니다.

### 10.2 V2 결과

Source: `notebooks_v2/results_v2.csv`

| Model | PR_AUC | ROC_AUC | Recall | Precision | F1 |
|---|---:|---:|---:|---:|---:|
| XGB_v2_tuned | 0.2959 | 0.9687 | 0.8699 | 0.0861 | 0.1566 |
| Ensemble_v2_best | 0.2928 | 0.9688 | 0.9978 | 0.0134 | 0.0264 |
| LGBM_v2_tuned | 0.2636 | 0.9666 | 0.8907 | 0.0734 | 0.1356 |
| XGB_v2_base | 0.2418 | 0.9616 | 0.8911 | 0.0600 | 0.1124 |
| LGBM_v2_base | 0.2246 | 0.9568 | 0.8831 | 0.0551 | 0.1037 |
| MLP_v2_plain_BCE | 0.1982 | 0.9409 | 0.0000 | 0.0000 | 0.0000 |
| MLP_v2_focal_batchnorm | 0.1953 | 0.9476 | 0.1633 | 0.3306 | 0.2187 |
| MLP_v2_focal | 0.1952 | 0.9443 | 0.1651 | 0.3391 | 0.2221 |
| MLP_v2_SMOTE | 0.1924 | 0.9469 | 0.5440 | 0.1531 | 0.2389 |
| MLP_v2_focal_dropout | 0.1850 | 0.9427 | 0.1067 | 0.3691 | 0.1655 |
| MLP_v2_weighted | 0.1766 | 0.9464 | 0.8588 | 0.0515 | 0.0972 |
| TabNet_v2 | 0.1016 | 0.9238 | 0.9150 | 0.0279 | 0.0541 |

V2에서는 팀원 파생변수와 추가 피처를 반영한 뒤 XGBoost 성능이 크게 좋아졌습니다. 하지만 수업이 딥러닝응용이므로, 여기서 머신러닝 모델을 더 올리기보다 V3에서 딥러닝 모델만 집중 개선했습니다.

### 10.3 V3 딥러닝 심화 결과

Source: `notebooks_v3/results_v3_dl.csv`

| Model | PR_AUC | ROC_AUC | Recall | Precision | F1 | notes |
|---|---:|---:|---:|---:|---:|---|
| MLP_V3 | 0.2305 | 0.9586 | 0.8774 | 0.0843 | 0.1538 | ResidualMLP+Wide, WeightedSampler, Focal(Optuna), PR_AUC early stopping |
| TabNet_V3 | 0.1192 | 0.9204 | 0.8725 | 0.0358 | 0.0688 | TabNet Optuna tuning |

V3의 핵심 성과는 MLP 개선입니다.

| 비교 | PR_AUC |
|---|---:|
| 기존 최고 MLP 계열 | 약 0.198 |
| MLP_V3 | 0.2305 |
| 개선폭 | 약 +0.033 |
| XGB_v2_tuned | 0.2959 |
| XGB_v2_tuned와 MLP_V3 격차 | 약 0.065 |

즉, 딥러닝 모델이 최종 1등은 아니지만, V3에서 기존 MLP보다 확실히 좋아졌습니다.

---

## 11. Threshold 분석

V2 최고 모델인 `XGB_v2_tuned`는 threshold를 바꾸면 성격이 크게 달라집니다.

Source: `notebooks_v2/06_Threshold_Optimization_V2.ipynb`

| threshold | Recall | Precision | F1 | 경고 비율 |
|---:|---:|---:|---:|---:|
| 0.3 | 0.927 | 0.057 | 0.108 | 10.8% |
| 0.5 | 0.870 | 0.086 | 0.157 | 6.8% |
| 0.7 | 0.783 | 0.128 | 0.220 | 4.1% |
| F1 최적 0.95 | 0.359 | 0.330 | 0.344 | 0.7% |
| F2 최적 0.87 | 0.591 | 0.206 | 0.305 | 1.9% |

해석:

- threshold를 낮추면 백오더를 많이 잡지만 경고가 많아집니다.
- threshold를 높이면 경고는 줄지만 놓치는 백오더가 늘어납니다.
- 그래서 모델을 고를 때는 PR_AUC를 보고, 실제 운영 기준은 threshold 분석으로 정합니다.

V3 MLP도 threshold를 조정하면 F1이 크게 좋아졌습니다.

Source: `notebooks_v3/01_MLP_V3_DL_Imbalance.ipynb`

| 기준 | threshold | Recall | Precision | F1 |
|---|---:|---:|---:|---:|
| 기본 기준 | 0.50 | 0.877 | 0.084 | 0.154 |
| F1 최적 | 0.88 | 0.400 | 0.265 | 0.319 |
| F2 최적 | 0.80 | 0.658 | 0.173 | 0.274 |

즉, `threshold = 0.5`만 보고 모델을 판단하면 안 됩니다. 백오더를 많이 잡고 싶은지, 경고 수를 줄이고 싶은지에 따라 기준점이 달라집니다.

---

## 12. 현재 결론

### 12.1 성능 기준 결론

현재 전체 최고 모델은 `XGB_v2_tuned`입니다.

| 순위 | 모델 | PR_AUC | 해석 |
|---:|---|---:|---|
| 1 | XGB_v2_tuned | 0.2959 | 전체 최고 성능 |
| 2 | Ensemble_v2_best | 0.2928 | PR_AUC는 높지만 threshold 0.5에서 Precision이 매우 낮음 |
| 3 | LGBM_v2_tuned | 0.2636 | 강한 트리 기반 기준 모델 |
| 4 | MLP_V3 | 0.2305 | 현재 최고 딥러닝 모델 |
| 5 | TabNet_V3 | 0.1192 | 튜닝했지만 성능은 낮음, 해석 보조 가치 |

### 12.2 딥러닝응용 수업 관점 결론

딥러닝 모델이 최종 1등은 아닙니다. 하지만 이를 단순 실패로 해석하기는 어렵습니다.

이 프로젝트에서는 다음을 확인했습니다.

1. 기본 MLP는 XGBoost보다 낮은 성능을 보였습니다.
2. Focal Loss, 불균형 샘플링, Residual 구조, Optuna 튜닝을 적용하면 MLP 성능이 실제로 개선되었습니다.
3. MLP_V3는 기존 MLP보다 PR_AUC를 약 +0.033 개선했습니다.
4. TabNet은 정형 데이터 특화 모델이지만, 이 데이터에서는 XGBoost나 MLP_V3보다 낮았습니다.
5. 정형 테이블 데이터에서는 여전히 GBDT 계열 모델이 강했습니다.

보고서에서는 다음과 같이 정리할 수 있습니다.

> 본 프로젝트는 백오더 예측이라는 희귀 이벤트 분류 문제를 다루었다.  
> 팀원 파생변수와 안전한 전처리를 반영한 V2에서 XGBoost가 가장 높은 성능을 보였다.  
> 이후 딥러닝응용 수업의 목적에 맞춰 V3에서는 머신러닝 모델을 추가 튜닝하지 않고 딥러닝 모델만 개선했다.  
> 그 결과 Residual MLP, WeightedRandomSampler, Focal Loss, Optuna, PR_AUC 기준 Early Stopping을 적용한 MLP_V3가 기존 MLP보다 뚜렷하게 향상되었다.  
> 다만 최종적으로는 XGBoost가 가장 높았으며, 이는 정형 테이블 데이터에서 트리 기반 모델이 강한 특성을 보인 결과로 해석된다.

---

## 13. GitHub 관리 정책

GitHub에는 코드와 결과 요약 파일 중심으로 관리합니다.

다음은 용량 또는 개인 참고용 성격 때문에 GitHub에 올리지 않습니다.

| 항목 | 이유 |
|---|---|
| `dataset_1/` | 원본 데이터는 용량이 크고 재배포 이슈가 있을 수 있습니다. |
| `processed/` | 전처리 산출물은 노트북으로 재생성 가능합니다. |
| `processed_v2/` | V2 전처리 산출물입니다. 용량이 크므로 GitHub에는 올리지 않습니다. |
| `team_chunsang_v1/dataset_v1/` | 박천상 팀원 참고 노트북의 원본 데이터입니다. |
| `*.pt`, `*.pth` | 모델 가중치 파일입니다. 용량이 커질 수 있습니다. |
| `*.npy` | 예측 확률 저장 파일입니다. 노트북 실행으로 재생성 가능합니다. |
| `기타 파일/` | 개인 참고 문서와 팀 공유용 임시 자료입니다. |

공유해야 하는 핵심 파일은 다음입니다.

| 항목 | 공유 이유 |
|---|---|
| `README.md` | 프로젝트 전체 설명 문서 |
| `notebooks_v1/` | 1차 실험 재현 |
| `notebooks_v2/` | 팀원 파생변수 기반 V2 실험 재현 |
| `notebooks_v3/` | 딥러닝 전용 심화 실험 재현 |
| `notebooks_v1/results.csv` | 1차 실험 결과 |
| `notebooks_v2/results_v2.csv` | V2 결과 |
| `notebooks_v3/results_v3_dl.csv` | V3 딥러닝 결과 |

---

## 14. 실행 순서

원본 데이터가 로컬에 준비되어 있다는 전제에서 다음 순서로 실행합니다.

### 1차 실험

```text
notebooks_v1/02_Preprocess.ipynb
notebooks_v1/03_Baseline_Tree.ipynb
notebooks_v1/04_MLP.ipynb
notebooks_v1/05_FT_Transformer.ipynb
notebooks_v1/06_TabNet.ipynb
notebooks_v1/07_AutoEncoder.ipynb
notebooks_v1/08_Decision_Analysis.ipynb
notebooks_v1/09_SMOTE.ipynb
notebooks_v1/10_AE_Hybrid.ipynb
```

### V2 실험

```text
notebooks_v2/01_Preprocess_V2.ipynb
notebooks_v2/02_Baseline_Tree_V2.ipynb
notebooks_v2/03_MLP_V2.ipynb
notebooks_v2/04_TabNet_V2.ipynb
notebooks_v2/05_Ensemble_V2.ipynb
notebooks_v2/06_Threshold_Optimization_V2.ipynb
```

### V3 딥러닝 심화 실험

```text
notebooks_v3/01_MLP_V3_DL_Imbalance.ipynb
notebooks_v3/02_TabNet_V3_Tuning.ipynb
```

V3는 `processed_v2/` 데이터를 사용하므로, 먼저 `notebooks_v2/01_Preprocess_V2.ipynb`를 실행해야 합니다.

---

## 15. 최종 한 줄 결론

**V2에서 팀원 파생변수와 안전한 전처리를 반영해 XGBoost가 최고 성능을 보였고, V3에서는 딥러닝 전용 기법을 적용해 MLP 성능을 의미 있게 끌어올렸지만, 정형 테이블 데이터 특성상 최종 최고 성능은 XGBoost가 유지되었습니다.**
