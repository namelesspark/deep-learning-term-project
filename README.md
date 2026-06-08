# Deep Learning Application Term Project

Backorder prediction project for the Deep Learning Application course.

Repository: <https://github.com/namelesspark/deep-learning-term-project.git>

## Project Goal

백오더(`went_on_backorder`) 발생 가능성을 예측하는 분류 문제입니다.  
데이터에서 백오더 Yes 비율이 매우 낮기 때문에 단순 accuracy는 주요 지표로 사용하지 않습니다.

주요 비교 지표:

- `PR_AUC`: 백오더 위험 상품을 얼마나 잘 위쪽에 줄 세우는지 보는 지표
- `Recall`: 실제 백오더 중 모델이 잡아낸 비율
- `Precision`: 백오더라고 예측한 것 중 진짜 백오더인 비율
- `F1`: Precision과 Recall의 균형

## Repository Policy

GitHub에는 노트북, 코드, 결과표만 관리합니다.  
아래 파일은 용량 또는 재생성 가능성 때문에 저장소에 올리지 않습니다.

- 원본 데이터: `dataset_1/`
- 전처리 결과: `processed/`
- v1 원본 데이터: `v1/dataset_v1/`
- 모델 가중치와 예측 결과: `*.pt`, `*.pth`, `*.npy`
- 개인 참고 문서: `기타 파일/`

팀원이 실행하려면 Kaggle 원본 CSV를 각자 `dataset_1/`에 넣고 `02_Preprocess.ipynb`부터 실행하면 됩니다.

## Notebook Flow

| Notebook | Purpose |
|---|---|
| `02_Preprocess.ipynb` | 공통 전처리, 파생변수 생성, train/val/test 저장 |
| `03_Baseline_Tree.ipynb` | LightGBM, XGBoost 기준선 |
| `04_MLP.ipynb` | MLP와 불균형 대응 실험 |
| `05_FT_Transformer.ipynb` | 정형 데이터용 Transformer |
| `06_TabNet.ipynb` | TabNet 모델 |
| `07_AutoEncoder.ipynb` | AutoEncoder 이상탐지 |
| `08_Decision_Analysis.ipynb` | 통합 비교, threshold, 비용 민감 분석 |
| `09_SMOTE.ipynb` | SMOTE 적용 비교 |
| `10_AE_Hybrid.ipynb` | AE score를 추가한 hybrid 실험 |

## Current Model Results

Source: `notebooks/results.csv`

All threshold-based metrics below use `threshold = 0.5`.

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

## Current Interpretation

### PR_AUC 기준

현재 전체 순위화 성능은 `XGBoost`가 가장 좋습니다.

`XGB_AE_hybrid`는 XGBoost보다 약간 낮지만 비슷한 수준입니다.  
`LightGBM`도 강한 기준선 역할을 합니다.

### threshold 0.5 기준 F1

`XGB_SMOTE`와 `MLP_SMOTE`는 F1이 높게 나옵니다.

다만 이 결과는 threshold 0.5에서 자른 운영점 기준입니다.  
PR_AUC는 오히려 기본 XGBoost보다 낮기 때문에, SMOTE가 전체 순위화 능력을 개선했다고 보기는 어렵습니다.

요약:

- `XGBoost`: 전체적으로 백오더 위험 상품을 가장 잘 줄 세움
- `XGB_SMOTE`: threshold 0.5 기준 F1은 좋지만 PR_AUC는 하락
- `MLP_3_focal`: 딥러닝 모델 중 PR_AUC가 가장 좋음
- `MLP_SMOTE`: 딥러닝 모델 중 threshold 0.5 기준 F1이 좋음
- `FT_Transformer`, `TabNet`: 현재 결과로는 MLP/트리보다 낮음

## Why PR_AUC Is Used

백오더는 Yes가 매우 적은 불균형 문제입니다.  
이 경우 모델 선택 단계에서는 threshold 0.5에서 한 번 자른 결과보다, 백오더 위험 상품을 전체적으로 얼마나 잘 줄 세웠는지가 중요합니다.

그래서 모델 자체의 성능 비교는 `PR_AUC`를 중심으로 보고, 실제 운영 기준은 `08_Decision_Analysis.ipynb`에서 threshold를 조정해 판단합니다.

## Next Steps

1. `08_Decision_Analysis.ipynb`에서 threshold 0.5가 아닌 최적 threshold 비교
2. 비용 민감 threshold 분석
3. PR_AUC 기준 모델 선택과 F1 기준 운영점 차이를 보고서에 정리
4. v1 파생변수 방식과 현재 SSOT 전처리 방식 비교 서술
