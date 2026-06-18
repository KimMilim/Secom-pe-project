# SECOM 반도체 불량 분류 프로젝트

## 프로젝트 목적
SK하이닉스 Product Engineering(PE) 직무 포트폴리오용 프로젝트.
공개 반도체 공정 데이터(SECOM)로 수율 불량 예측 모델을 구현하고,
어떤 공정 변수가 불량에 영향을 주는지 해석하는 것이 핵심 목표.

---

## 데이터셋
- **출처**: UCI ML Repository - SECOM Dataset
- **구성**: 1,567개 샘플, 591개 공정 센서 피처, 불량/정상 이진 레이블
- **특징**: 결측치 다수 존재, 불량 비율 약 6% (극심한 클래스 불균형)

---

## 개발 환경
- Python 3.x, MacBook Pro (로컬 학습)
- 주요 라이브러리: pandas, numpy, scikit-learn, xgboost, imbalanced-learn, shap, matplotlib, seaborn

---

## 프로젝트 구조
```
secom-pe-project/
├── data/
│   ├── SECOM.data         # 원본 피처 데이터
│   └── SECOM_labels.data  # 레이블 데이터
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   └── 04_shap_analysis.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── evaluate.py
├── outputs/
│   └── figures/           # 시각화 결과물
├── CLAUDE.md
└── README.md
```

---

## 작업 지시 원칙

### 코드 작성 시
- 모든 코드는 **VSCode + .ipynb** 기준으로 작성
- 각 셀마다 한국어 주석으로 무엇을 하는지 설명
- 중간 결과는 반드시 print 또는 시각화로 출력

### 평가 지표
- **Accuracy는 사용하지 않음** (불균형 데이터이므로 의미 없음)
- 주요 지표: **Recall, F1-score, ROC-AUC**
- 불량(Positive)을 놓치지 않는 것이 최우선 → Recall 중심

### 모델링 우선순위
1. Random Forest (베이스라인)
2. XGBoost (메인 모델)
3. LightGBM (선택)

### SHAP 분석 필수 포함
- 전체 피처 중 상위 20개 중요도 시각화 (bar plot)
- 개별 예측 설명 (waterfall plot) 최소 1개
- 결과 해석 시 "공정 관점"으로 설명하는 주석 추가

---

## 각 단계별 체크리스트

### Step 1. EDA
- [x] 데이터 shape, dtypes 확인
- [x] 결측치 비율 컬럼별 계산 및 시각화
- [x] 클래스 분포 확인 (불량 vs 정상 비율)
- [x] 피처별 분포 히스토그램

### Step 2. 전처리
- [x] 결측치 50% 이상 컬럼 제거
- [x] 나머지 결측치 median imputation
- [x] 분산 0인 피처 제거 (VarianceThreshold)
- [x] SMOTE로 클래스 불균형 처리
- [x] Train/Test split (8:2)

### Step 3. 모델링
- [x] Random Forest 학습 및 평가
- [x] XGBoost 학습 및 평가
- [x] Classification report 출력
- [x] Confusion matrix 시각화
- [x] ROC Curve 비교
- [x] (보강) 결정 임계값 튜닝 — 0.5에서 Recall 0 문제를 CV로 해결

### Step 4. SHAP 분석
- [x] shap.TreeExplainer 적용
- [x] Summary plot (상위 20 피처) — bar + beeswarm
- [x] Waterfall plot (TP 샘플 1개)
- [x] 상위 피처 해석 주석 작성 (공정 관점)
- [x] 포트폴리오용 인사이트 문장 도출

---

## 포트폴리오 최종 메시지 (코드 작성 시 항상 염두)
> "단순히 모델 정확도를 높이는 것이 아니라,
> 어떤 공정 변수가 수율 불량에 영향을 주는지 해석하는 데 집중했습니다."

이 문장을 뒷받침할 수 있는 결과물을 만드는 것이 이 프로젝트의 핵심.

---

## 작업 진행 로그
- **Step 1. EDA 완료** — `notebooks/01_eda.ipynb`
  - 1,567 × 590, 불량 6.64%(불균형 14.1:1), 결측치 50%↑ 28개, 분산 0 컬럼 116개
  - 결과 그림: `outputs/figures/01_class_distribution.png`, `02_missing_values.png`, `03_feature_distributions.png`
- **Step 2. 전처리 완료** — `notebooks/02_preprocessing.ipynb`
  - split 먼저(8:2 stratify, seed=42) → 모든 변환 train fit (누수 방지)
  - 결측50%↑ 24개 + 분산0 116개 제거 → 최종 **450 피처**
  - SMOTE(train만) 1170:1170, test는 불량 6.69% 유지
  - 저장: `data/processed/secom_processed.npz` (X_train/y_train, X_train_clean/y_train_clean[SMOTE전], X_test/y_test, feature_names)
  - 그림: `outputs/figures/04_smote_balance.png`
- **Step 3. 모델링 완료** — `notebooks/03_modeling.ipynb`
  - RF·XGB 학습. **기본 임계값 0.5에서 두 모델 모두 Recall=0** (불량 확률이 0.5 미달)
  - SMOTE를 CV 파이프라인에 넣어 누수 없이 F1-최대 임계값 선택 (RF 0.32 / XGB 0.12)
  - 튜닝 후 test: RF Recall 0.333 / F1 0.311 / **ROC-AUC 0.805** (선택), XGB Recall 0.333 / F1 0.280 / AUC 0.724
  - 저장: `outputs/models/{random_forest,xgboost}.joblib` (각각 {model, threshold} dict)
  - 그림: `05_confusion_matrix.png`, `06_roc_curve.png`, `07_threshold_tuning.png`
- **Step 4. SHAP 분석 완료** — `notebooks/04_shap_analysis.ipynb` ★ 프로젝트 핵심
  - 선택 모델(RF) TreeExplainer → 불량 기여도 상위 20 센서 도출 (TOP: sensor_519/095/511/247/059…)
  - bar + beeswarm(방향성) + waterfall(TP #249, 불량확률 0.40) + 공정 관점 해석
  - 그림: `08_shap_summary_bar.png`, `09_shap_beeswarm.png`, `10_shap_waterfall.png`
  - 폰트 주의: SHAP waterfall은 음수에 유니코드 MINUS(U+2212) 사용 → AppleGothic엔 글리프 없어 □. 이 노트북만 `Arial Unicode MS`(한글+MINUS 모두 포함) 사용.
- **Step 5. 센서→공정 매핑 시나리오 완료** — `notebooks/05_process_mapping.ipynb`
  - 영향 방향(값↑→불량↑/↓)을 SHAP-값 상관으로 데이터 기반 산출, 가정 공정명 매핑 + SPC 관리 조치 제안
  - 산출: `outputs/process_mapping_scenario.csv`, 그림 `11_process_mapping.png`. ※ 공정명은 전부 가정(시연용)
- **Step 6. 하이퍼파라미터 튜닝 완료** — `notebooks/06_hyperparameter_tuning.ipynb`
  - SMOTE 파이프라인 + RandomizedSearchCV(F1, cv=4, n_iter=20), 누수 없이 임계값 재선택
  - 튜닝 RF Recall 0.333→**0.571**(class_weight=balanced), 튜닝 XGB F1 0.280→0.324
  - 저장: `outputs/models/{random_forest_tuned, xgboost_tuned}.joblib`
- **Step 7. LightGBM 추가 & 3-모델 비교 완료** — `notebooks/07_lightgbm_comparison.ipynb`
  - LightGBM 동일 절차 튜닝(Recall 0.524). 최종 비교 → **추천: RandomForest(튜닝)** (Recall 0.571 최고)
  - 저장: `lightgbm_tuned.joblib`, `final_model.joblib`(=튜닝 RF), 그림 `12_model_comparison.png`
- **Step 8. 포트폴리오 요약 완료** — `notebooks/08_portfolio_summary.ipynb`
  - 전체 모델 성능 집계 + SHAP 핵심 센서 재집계 + 발표용 내러티브
  - 1장 슬라이드 자동 생성 → `outputs/portfolio_slide.png` (16:9)
  - README에 슬라이드·결과 표·핵심 그림 임베드 완료

## 환경 메모 (macOS)
- 가상환경 `.venv/` (Python 3.9.6), Jupyter 커널 `secom-pe` ("Python (secom-pe)")
- XGBoost 사용 시 OpenMP 런타임 필요 → `brew install libomp`
- matplotlib 한글 폰트는 `AppleGothic`. **주의**: `sns.set_style()`이 폰트를 덮어쓰므로
  set_style을 먼저 호출한 뒤 `mpl.rcParams['font.family']='AppleGothic'`를 설정할 것.
- 실제 데이터 파일명은 소문자 `data/secom.data`, `data/secom_labels.data` (UCI 원본)
