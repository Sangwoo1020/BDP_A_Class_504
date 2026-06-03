# 🧥 중고 의류 컨디션 분류 모델

> **이미지 기반 중고 의류 상태 자동 예측 시스템**  
> 2026-1 빅데이터프로그래밍 A분반 | Team 504

---

## 프로젝트 개요

국내 중고 의류 시장은 연평균 16.5% 성장하고 있지만, 현재 중고 플랫폼에서의 의류 컨디션 평가는 **판매자 개인의 주관적 판단**에 의존합니다. 이로 인해 구매자는 실제 상태를 확인하기 어렵고, 품질 불확실성을 안고 거래해야 합니다.

본 프로젝트는 **의류 이미지만으로 컨디션을 1~5등급으로 자동 예측**하는 딥러닝 모델을 설계하고 검증합니다.

---

## 팀 구성

| 이름 | 역할 |
|------|------|
| 김동석 (팀장) | Project Lead · 전체 일정 조율 · 보고서 및 발표자료 제작 · 실험 결과 정리 |
| 김상우 | Baseline & Evaluation · Concat-MLP 베이스라인 구현 · 평가 코드 작성 |
| 노진수 | Data Pipeline · 데이터 전처리 및 임베딩 · HCF 148차원 추출 파이프라인 구현 |
| 민재영 | Experiment & Tuning · 실험 A~D 전체 학습 실행 · FiLM 융합·곱항 결합(v4) 구조 개선 · Ablation Study 실행 |
| 유성재 | Documentation & Presentation · HCFEncoder 설계 구현 · 멀티뷰 결합 강화 구조 설계 · 발표자료 제작 |

---

## 연구 목표 및 가설

### 최종 목표
중고 거래 플랫폼에서 활용 가능한 의류 상태 평가 시스템 검증

### 연구 가설

| 가설 | 연구 질문 |
|------|-----------|
| H1 — 이미지 기반 예측 | 중고 의류 이미지만으로 의류 상태를 신뢰성 있게 분류할 수 있는가? |
| H2 — 전역 + 국소 특징 | Local-Global 특징을 함께 고려할 때 모델 분류 성능이 향상되는가? |
| H3 — 소재·직물 특성 | 의류 소재 또는 직물 특성을 함께 고려할 경우 정확도가 향상되는가? |

---

## 데이터셋

**Zenodo 중고 의류 데이터셋** ([링크](https://zenodo.org/records/13788681))

| 항목 | 내용 |
|------|------|
| 이미지 수 | 31,997쌍 (전면·후면 이미지 + 라벨) |
| 타겟 변수 | condition 1~5등급 |
| 보조 속성 | pilling(보풀), stains(오염), holes(구멍) 등 |

### 전처리 파이프라인

```
원본 데이터
    ↓ 전·후면 Pair 구성
의류 영역 정제 (RGB → HSV 변환 → Saturation Mask, threshold=25)
    ↓ Morphology (Close + Open) 노이즈 제거
Bounding Box Crop
    ↓ Pad Resize (종횡비 왜곡 방지)
300×300 Resize (EfficientNet-B3 권장 해상도)
    ↓ ImageNet 정규화 (mean/std)
Train / Val / Test 분할 (7:1.5:1.5)
```

> fallback 비율: **13.1% → 0.1%** 로 대폭 감소 (2단계 threshold 재검출 적용)

---

## 모델 설계

### 핵심 아이디어

의류 이미지에서 **전역 특징(전체적인 사용감)** 과 **국소 특징(얼룩·보풀·마모)** 을 동시에 추출하고, GLAM(Global-Local Attention Module) 기반 Fusion으로 통합하여 컨디션을 판단합니다.

### 선행 연구 착안점

| 연구 | 착안점 |
|------|--------|
| TextileNet (2023) | 소재별 특성 라벨을 컨디션 분류에 적용 |
| GLAM — Global-Local Attention (WACV 2022) | GLAM 모듈을 의류 컨디션 분류에 차용 |
| Defect Detection 연구 | EfficientNet-B3를 주요 Backbone으로 선정 |

### Backbone 비교

| | ResNet-50 | EfficientNet-B3 |
|--|-----------|-----------------|
| 파라미터 수 | ~25M | ~12M (더 경량) |
| 특징 채널 | 2048 ch → GAP → FC | 1536 ch → 10×10 feature map |
| 핵심 구조 | Skip Connection + BatchNorm | Compound Scaling + SE Block + Swish |
| 입력 해상도 | 224×224 | 300×300 |
| 역할 | Primary Baseline | 실험 A~D 공통 Backbone |
| 평가 | Condition 분류 한계 → 최종 제외 | Fine-grained 분류에 적합 → 채택 |

### 전이학습 전략: 2단계 Full Fine-tuning

```
1단계 — Warmup (3 epoch)
  · Backbone Freeze
  · 학습 대상: GLAM 이후 모듈 + MLP Head
  · Head LR: 3e-4

2단계 — Full Fine-tuning
  · Backbone 전체 UnFreeze
  · Backbone LR: 3e-5 / Head LR: 3e-4 (differential)
  · Optimizer: AdamW (Weight Decay: 1e-4)
  · Loss: CrossEntropyLoss (weighted + label smoothing 0.05)
  · Batch: 128 / Max Epoch: 100 / Early Stopping: patience=10 (val loss 기준)
```

### GLAM — 4가지 Attention 구조

GLAM은 **Local(채널·공간) + Global(채널·공간)** 4가지 어텐션을 병렬로 적용합니다.

| 종류 | 역할 |
|------|------|
| Local Channel Attention (LCA) | 채널별 손상 중요도 강조 (ECA 방식) |
| Local Spatial Attention (LSA) | 위치별 손상 패턴 강조 (Dilated Conv 1,2,3) |
| Global Channel Attention (GCA) | 채널 간 전반적 관계 모델링 (Self-Attention) |
| Global Spatial Attention (GSA) | 공간 위치 간 전반적 관계 모델링 (Self-Attention) |

```
F_out = F + α · Local_feature + β · Global_feature
(학습 가능한 α, β로 Local/Global 비중 자동 조절, 원본 F는 잔차로 보존)
```

---

## 실험 설계 (Experiment A → D)

각 실험은 **하나의 변수만 변경**하여 구조별 기여도를 순수하게 분리합니다.

### 실험 구조 요약

```
A. EfficientNet-B3 단일뷰  ← Primary Baseline
   앞면 이미지 → EfficientNet-B3 → GAP → FC(1536→512→5) → CE Loss

B. EfficientNet-B3 멀티뷰  [A vs B: 멀티뷰 효과]
   앞면+뒷면 → EfficientNet-B3 (Weight 공유) → GAP → Concat(3072) → FC(3072→512→5) → CE Loss

C. EfficientNet-B3 + GLAM  [B vs C: Attention 효과]
   앞면+뒷면 → EfficientNet-B3 → GLAM(4-way, Weight 공유) → GAP → Concat(3072) → MLP(3072→512→5) → CE Loss

D. EfficientNet-B3 + GLAM + HCF  [C vs D: 멀티모달 효과]
   앞면+뒷면 → EfficientNet-B3 → GLAM → GAP → Concat(3072)  ─┐
   HCF(148차원) → HCFEncoder(148→128→64)                      ─┤
                                          Final Concat(3136) → MLP(3136→1024→256→16→5) → CE Loss
```

### HCF (Handcrafted Features)

이미지 feature만으로 구분하기 어려운 상태 정보를 보완하는 **정형 feature**입니다.

| 버전 | 차원 | 구성 | 특징 |
|------|------|------|------|
| D-Meta | 148차원 | category, type, color, texture 등 객관 속성 | 배포 가능 (누수 없음) |
| D-Full | 183차원 | Meta + pilling, stains, smell 등 손상 상태 | 성능 상한선 측정용 |

### 전체 프레임워크 흐름 (Experiment D)

```
Front Image (B, 3, 300, 300) ─┐
                               ├─ EfficientNet-B3 (shared weights, pretrained)
Back Image  (B, 3, 300, 300) ─┘
        ↓
Feature map: (B, 1536, 10, 10) × 2
        ↓
GLAM (4-way Attention, shared weights)
  LCA + LSA (Local)
  GCA + GSA (Global)
  F_out = F + α·F_L + β·F_G
        ↓
GAP → Front/Back Vector (B, 1536) × 2
        ↓
Concatenate (B, 3072)  ←── Image Feature
        +
HCF (B, 148) → HCFEncoder [BN→Linear(148→128)→GELU→Dropout→Linear(128→64)→GELU→Dropout]
                              ↓ HCF Embedding (B, 64)
        ↓
Final Concatenate (B, 3136)
        ↓
MLP: 3136 → 1024 → 256 → 16 → 5 logits
        ↓
CrossEntropyLoss → Condition Class (1~5)
```

---

## 실험 결과

### 모델별 성능 비교 (5 Runs, Seed = 41~45)

| 모델 | 구조 | Accuracy | F1-Macro | 결과 요약 |
|------|------|----------|----------|-----------|
| A | EfficientNet-B3 단일뷰 | 0.3944 | 0.2902 | 단일 이미지 기반, 성능 정체 |
| B | EfficientNet-B3 멀티뷰 | 0.3980 | 0.2910 | 멀티뷰 추가 효과 미미 |
| C | EfficientNet-B3 + GLAM | 0.4003 | 0.2673 | Acc 소폭 상승, F1-Macro 하락 |
| D-Meta | GLAM + HCF(Meta) | 0.4965 | 0.4199 | HCF 추가 후 두 지표 모두 대폭 상승 |
| D-Full | GLAM + HCF(Full) | 0.5439 | 0.4922 | Full HCF 적용 시 최고 성능 달성 |

> **핵심 해석**: A~C는 Accuracy & F1 모두 39~40% / 0.26~0.29로 수렴 → 이미지 구조와 무관하게 성능 한계 존재  
> D에서 HCF 추가 시 두 지표 모두 +10~14%p 동반 상승 → **메타데이터가 핵심 변수**

### Ablation Study — Model D(Full) 구조별 기여도

| 모델 버전 | 변경 내용 | ACC | F1-Macro | Kappa |
|----------|----------|-----|----------|-------|
| V2 | 단순 Concat 결합 | 0.544 | 0.510 | 0.417 |
| V3 | 차이항 + HCF FiLM + GLAM | 0.613 | 0.510 | 0.476 |
| V4 | V2 + 곱항 interaction 추가 | **0.620** | **0.519** | **0.486** |

> V4 이후 정규화 강화(V6), warmup 단축(V7), GLAM 제거(V8) 실험에서 지표 개선 제한적  
> → **성능 향상의 핵심은 attention이나 학습 스케줄보다 feature interaction 결합 방식**

---

## 코드 구조

```
BDP_A_Class_504/
├── data_preprocessing_pipeline.py   # 전처리 파이프라인 (HSV Crop, Resize, Split)
├── exp_A_singleview_baseline.ipynb  # Experiment A: EfficientNet-B3 단일뷰
├── exp_B_multiview_concat.ipynb     # Experiment B: 멀티뷰 + Concat-MLP
├── exp_C_glam_attention.ipynb       # Experiment C: 멀티뷰 + GLAM 4-way Attention
├── exp_D_hcf_meta.ipynb             # Experiment D: GLAM + HCF Meta (148차원)
└── exp_D_hcf_full.ipynb             # Experiment D: GLAM + HCF Full (183차원)
```

### 실행 방법

```bash
# 1. 전처리
python data_preprocessing_pipeline.py \
    --data_dir /path/to/raw_data \
    --output_dir /path/to/preprocessed \
    --target_size 300

# 2. 실험 실행 (Jupyter Notebook)
jupyter notebook exp_A_singleview_baseline.ipynb

# 환경변수로 seed, run index 지정 가능
SEED=42 RUN_INDEX=0 jupyter notebook exp_A_singleview_baseline.ipynb
```

### 주요 의존성

```
torch >= 2.0
timm
safetensors
scikit-learn
pandas
opencv-python
tqdm
matplotlib
seaborn
huggingface_hub
```

---

## 실험 환경

| 항목 | 내용 |
|------|------|
| Framework | PyTorch 2.x + timm |
| GPU | CUDA 지원 GPU 권장 (RTX 3090 이상) |
| Batch Size | 128 |
| Image Size | 300×300 |
| Random Seed | 41~45 (5 Runs) |
| AMP | bfloat16 (선택) |

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| **Accuracy** | 정확히 맞춘 비율 (주지표) |
| **F1-Macro** | 클래스별 F1 평균 — 클래스 불균형 보정 (보조지표) |
| Balanced Accuracy | 클래스 불균형 보정 정확도 |
| Cohen's Kappa | 우연을 제외한 실질적 일치도 |
| MAE | 예측 등급과 실제 등급 간 평균 절대 오차 |

---

## 결론 및 시사점

### 학술적 시사점
1. **주관적 컨디션 판단의 정량화** — 판매자 주관 영역에 Computer Vision 적용 가능성 제시
2. **Global-Local Attention의 적용 가능성** — 전체적 사용감과 국소 손상을 동시 반영
3. **멀티모달 접근의 유효성 확인** — 이미지 + HCF 결합이 컨디션 분류 도메인에서 유효

### 실무적 시사점
1. **정보 비대칭 완화** — 객관적·일관적 컨디션 기준으로 구매 신뢰도 향상
2. **등록·가격 책정 간소화** — 이미지 업로드만으로 컨디션 자동 분류
3. **플랫폼 운영 효율 향상** — 인적 검수 비용 절감 및 표준화된 라벨 기반 검색·추천

### 한계점 및 향후 계획

| 한계 | 향후 계획 |
|------|-----------|
| GLAM 성능 향상 폭 제한 (해상도·품질 문제) | 디블러링·노이즈 제거 후 GLAM 재학습 |
| 시각 정보 위주 단일 모달 | 가격·브랜드·사용기간 등 메타데이터 확장 |
| 단일 플랫폼·단일 데이터셋 기반 | 당근·번개장터 등 복수 플랫폼 데이터로 검증 |

---

## 참고 문헌

1. Clothing Dataset for Second-Hand Fashion, Zenodo, 2024. https://zenodo.org/records/13788681
2. TextileNet: A Deep Learning Approach for Textile Fabric Material Identification. IEEE Xplore. https://ieeexplore.ieee.org/document/10441457
3. Garment Condition Assessment Using Computer Vision, Master thesis, Lund University, 2024.
4. Rocha, A. et al., "Using Object Detection Technology to Identify Defects in Garments," Sensors, 2023.
5. Tan, M. and Le, Q. "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks." ICML, 2019. https://proceedings.mlr.press/v97/tan19a.html
6. He, K. et al. "Deep Residual Learning for Image Recognition." CVPR, 2016. https://arxiv.org/abs/1512.03385
7. Shi, X. et al. "Deep Neural Networks for Rank-Consistent Ordinal Regression Based on Conditional Probabilities." 2023. https://arxiv.org/abs/2111.08851
8. Song, C. H. et al. "All the Attention You Need: Global-Local, Spatial-Channel Attention for Image Retrieval." WACV, 2022. https://arxiv.org/abs/2107.08000

---

> 한성대학교 | 2026-1 빅데이터프로그래밍 A분반 | Team 504  
> GitHub: https://github.com/Sangwoo1020/BDP_A_Class_504
