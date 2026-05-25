# 🧥 중고 의류 컨디션 분류 모델

> **이미지 기반 중고 의류 상태 자동 예측 시스템**  
> 2026-1 빅데이터프로그래밍 A분반 | Team 504

---

##  프로젝트 개요

국내 중고 의류 시장은 연평균 16.5% 성장하고 있지만, 현재 중고 플랫폼에서의 의류 컨디션 평가는 **판매자 개인의 주관적 판단**에 의존합니다. 이로 인해 구매자는 실제 상태를 확인하기 어렵고, 품질 불확실성을 안고 거래해야 합니다.

본 프로젝트는 **의류 이미지만으로 컨디션을 1~5등급으로 자동 예측**하는 딥러닝 모델을 설계하고 검증합니다.

---

##  팀 구성

| 이름 | 역할 |
|------|------|
| 김동석 (팀장) | Project Lead · 전체 일정 조율 · 보고서 작성 |
| 김상우 | Baseline & Evaluation · Concat-MLP 구현 · Grad-CAM 시각화 |
| 노진수 | Data Pipeline · 데이터 전처리 및 임베딩 파이프라인 |
| 민재영 | Experiment & Tuning · Ablation 실험(A~E) · 하이퍼파라미터 튜닝 |
| 유성재 | Documentation & Presentation · Co-attention / GMU Fusion 모듈 구현 |

---

##  연구 목표 및 가설

### 최종 목표
중고 거래 플랫폼에서 활용 가능한 의류 상태 평가 시스템 검증

### 연구 가설
| 가설 | 연구 질문 |
|------|-----------|
| H1 — 이미지 기반 예측 | 중고 의류 이미지만으로 의류 상태를 신뢰성 있게 분류할 수 있는가? |
| H2 — 소재·직물 특성 | 의류 소재를 함께 고려할 경우 정확도가 향상되는가? |
| H3 — 전역+국소 특징 | Local-Global 특징을 함께 고려할 때 모델 예측력이 향상되는가? |
| H4 — 서비스 적용 | 이미지 기반 모델이 중고 거래 플랫폼의 품질 불확실성 완화에 도움이 되는가? |

---

##  데이터셋

**Zenodo 중고 의류 데이터셋** ([링크](https://zenodo.org/records/13788681))

| 항목 | 내용 |
|------|------|
| 이미지 수 | 31,997쌍 (전면·후면·라벨) |
| 라벨 | condition 1~5등급 |
| 메타데이터 | 11개 속성 (pilling, stains, holes 등) |

### 전처리 파이프라인

```
원본 데이터
    ↓ HSV 변환 (채도 채널 분리)
의류 영역 정제 (Saturation Mask → threshold=25, fallback=12)
    ↓ Morphology (Close + Open)
Bounding Box Crop
    ↓ Pad Resize (종횡비 왜곡 방지)
224×224 (ResNet-50) / 300×300 (EfficientNet-B3)
    ↓ ImageNet 정규화
전·후면 Pair 구성 → Train / Val / Test 분할
```

> fallback 비율: **13.1% → 0.1%** 로 감소 (2단계 threshold 재검출 적용)

---

## 모델 설계

### 핵심 아이디어

의류 이미지에서 **전역 특징(전체적인 사용감)** 과 **국소 특징(얼룩·보풀·마모)** 을 동시에 추출하고, Attention 기반 Fusion으로 통합하여 컨디션을 판단합니다.

### 선행 연구 착안점

| 연구 | 착안점 |
|------|--------|
| TextileNet (2023) | ResNet-50을 Primary Baseline으로 채택 |
| GLAM — Global-Local Attention (WACV 2022) | GLAM 모듈을 의류 컨디션 분류에 차용 |
| Defect Detection 연구 | EfficientNet-B3를 주요 Backbone으로 선정 |

### Backbone 비교

| | ResNet-50 | EfficientNet-B3 |
|--|-----------|-----------------|
| 파라미터 수 | ~25M | ~12M (더 경량) |
| 특징 채널 | 2048 ch | 1536 ch → 10×10 feature map |
| 핵심 구조 | Skip Connection + Batch Norm | Compound Scaling + SE Block + Swish |
| 입력 해상도 | 224×224 | 300×300 |
| 역할 | Primary Baseline (실험 A) | 주요 제안 모델 (실험 B~D) |

### 전이학습 전략: Full Fine-tuning (2단계)

```
1단계 — Warmup (3 epoch)
  · Backbone Freeze
  · Head LR: 1e-3
  · 학습 대상: GLAM 이후 모듈 + Co-attn + MLP head

2단계 — Full Fine-tuning
  · Backbone UnFreeze
  · Backbone LR: 1e-5 / Head LR: 3e-4 (30배 차등)
  · Optimizer: AdamW (Weight Decay: 1e-4)
  · Loss: CORN (Conditional Ordinal Regression)
  · Early Stopping: patience=5 (val loss 기준)
```

---

##  실험 설계 (Study A → D)

각 실험은 **하나의 변수만 변경**하여 구조별 기여도를 순수하게 분리합니다.

### 실험 구조 요약

```
A. ResNet-50 단일뷰
   앞면 이미지 → ResNet-50 → GAP → FC(2048→512→4) → CORN Loss

B. EfficientNet-B3 단일뷰  [A vs B: Backbone 효과]
   앞면 이미지 → EfficientNet-B3 → GAP → FC(1536→512→4) → CORN Loss

C. EfficientNet-B3 멀티뷰  [B vs C: 멀티뷰 효과]
   앞면+뒷면 → EfficientNet-B3 (Weight 공유) → GAP → Concat(3072) → FC → CORN Loss

D. EfficientNet-B3 + GLAM + Co-attention  [C vs D: Attention 효과]
   앞면+뒷면 → EfficientNet-B3 → GLAM(4-way) → Spatial Co-attn(4-head) → GAP → Concat(3072) → FC → CORN Loss
```

### 전체 프레임워크 흐름 (Study D)

```
Front Image (B,3,300,300) ─┐
                            ├─ EfficientNet-B3 (shared) ─ GLAM(4-way) ─ T_a (B,100,1536)
Back Image  (B,3,300,300) ─┘                             GLAM(4-way) ─ T_b (B,100,1536)
                                                                  ↓
                                               4-Head Spatial Cross-attention
                                                                  ↓
                                              GAP → Concat(t'_a, t'_b) → (B,3072)
                                                                  ↓
                                              MLP: 3072 → 512 → 4 logits
                                                                  ↓
                                              CORN Loss → Condition Score (1~5)
```

---

## 실험 결과

### 모델별 평균 성능 (5 Runs)

| 모델 | Metric | Mean | Std |
|------|--------|------|-----|
| A — ResNet-50 단일뷰 | MAE | 0.8352 | 0.0024 |
| | Accuracy | 0.3807 | 0.0025 |
| B — EfficientNet-B3 단일뷰 | MAE | 0.8225 | 0.0022 |
| | Accuracy | 0.3840 | 0.0037 |
| C — EfficientNet-B3 멀티뷰 | MAE | 0.8158 | 0.0018 |
| | Accuracy | 0.4064 | 0.0009 |

> 실험 D (GLAM + Spatial Co-attention) 결과는 현재 진행 중입니다.

### 전체 Run 순위 (Test Acc 기준, A/B 5회 + C 1회)

| Rank | Model | Run | Test Acc | Test F1 Macro | Test Kappa |
|------|-------|-----|----------|---------------|------------|
| 1 | A | run_02 | 0.4061 | 0.2764 | 0.1757 |
| 2 | A | run_05 | 0.4033 | 0.2710 | 0.1725 |
| 3 | B | run_05 | 0.4026 | 0.2902 | 0.1767 |
| 4 | A | run_04 | 0.4023 | 0.2755 | 0.1734 |
| 5 | B | run_03 | 0.4007 | 0.2926 | 0.1756 |

### 모델 C (멀티뷰) 상세 결과

| Split | Acc | Balanced Acc | F1 Macro | Kappa | Best Epoch |
|-------|-----|-------------|----------|-------|------------|
| Val | 0.3976 | 0.2938 | 0.2755 | 0.1657 | 5 |
| Test | 0.3924 | 0.2927 | 0.2778 | 0.1604 | 5 |

**Confusion Matrix (Test, Model C)**

|  | Pred 1 | Pred 2 | Pred 3 | Pred 4 | Pred 5 |
|--|--------|--------|--------|--------|--------|
| True 1 | 0 | 6 | 89 | 2 | 7 |
| True 2 | 0 | 48 | 339 | 38 | 68 |
| True 3 | 0 | 50 | 617 | 145 | 148 |
| True 4 | 0 | 19 | 280 | 374 | 89 |
| True 5 | 0 | 35 | 386 | 199 | 188 |

> 클래스 1을 전혀 예측하지 못하며, 전반적으로 클래스 3으로의 쏠림 현상이 존재합니다. 클래스 4가 상대적으로 가장 양호한 예측률(~49%)을 보입니다.

---

## 실험 환경

| 항목 | 내용 |
|------|------|
| OS | Ubuntu / Linux |
| GPU | RTX 4090 24GB × 2 |
| Framework | PyTorch 2.5.1+cu121 |
| Python | 3.10 |
| CUDA | 12.1 |
| Batch Size | 48 |
| Random Seed | 42 |
| AMP | False |
| Early Stop patience | 5 |

---

## 평가 지표

| 지표 | 설명 |
|------|------|
| **Accuracy** | 정확히 맞춘 비율 (다중 분류 표준 지표) |
| **MAE** | 예측 등급과 실제 등급 간 평균 절대 오차 |
| **Balanced Accuracy** | 클래스 불균형 보정 정확도 |
| **F1 Macro** | 클래스별 F1 평균 (불균형 데이터에 적합) |
| **Cohen's Kappa** | 우연을 제외한 실질적 일치도 |
| RMSE (보조) | 이상치 민감도 판단 |

---

## 프로젝트 마일스톤

| 마일스톤 | 내용 | 상태 |
|---------|------|------|
| M1 | 중고 의류 이미지 데이터 수집 및 구조 파악 | 완료 |
| M2 | 이미지 전처리 및 앞면/뒷면 이미지 쌍 구성 | 완료 |
| M3 | Backbone 모델 구현 및 확정 | 완료 |
| M4 | GLAM Attention Module 구현 | 완료 |
| M5 | Spatial Cross-Attention 구조 구현 | 완료 |
| M6 | 실험 수행 및 환경 설정 | 완료 |
| M7 | Multi-view Fusion + CORN Loss 기반 최종 모델 선정 | 완료 |
| M8 | Ablation Study를 통한 구성 요소별 성능 기여도 분석 | 진행 중 |
| M9 | 최종 모델 성능 평가 및 피드백 | 예정 |

### 향후 계획

**M8 — Ablation Study**
- Without Back View: 멀티뷰 입력 효과 검증
- Without Co-Attention: 앞뒤 상호 정보 교환 구조 기여도 확인
- Without GLAM: 국소 손상 특징 강조 모듈 효과 분석
- Without CORN Loss: 순서형 등급 분류 손실 함수 필요성 검증

**M9 — 확장 실험**
- Hand-crafted Feature 추가 (밝기, 색상 분포, 마모 영역 등)
- JSON Metadata 결합 (브랜드, 카테고리, 소재 등)
- Image + Metadata Fusion (멀티모달 구조)
- TextileNet 소재 임베딩 추가 실험 (H2 가설 검증)
- Error Case Analysis (Grad-CAM 시각화)

---

## 기대 효과 및 한계

### 활용 가능성
- 판매자 등록 시 상태 등급 자동 추천
- "AI 참고 등급" 표시로 구매 신뢰도 향상
- 상태별 필터 검색 기능 제공
- 상태 기반 적정 가격 제안 연동
- 검수 인력 부담 경감

### 잠재적 한계
- 라벨 자체의 주관성 (평가자 간 차이)
- 특정 촬영 환경에 편향될 가능성
- 이미지만으로는 촉감, 냄새, 안감 상태 반영 불가
- 등급 경계의 모호성
- 완전 자동화보다는 참고 도구로 활용

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
