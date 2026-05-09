---
title: >-
  [Paper Note] TALON: Test-time Adaptive Learning for On-the-Fly Category Discovery
description: >-
  [CVPR2026][Model Compression][On-the-Fly Category Discovery] This paper proposes TALON, the first test-time adaptive framework for On-the-Fly Category Discovery (OCD). By combining semantics-aware prototype updating, stable encoder adaptation, and margin-aware logit calibration, TALON operates directly in continuous feature space without hash encoding, substantially alleviating category explosion and significantly improving novel category discovery accuracy.
tags:
  - CVPR2026
  - Model Compression
  - On-the-Fly Category Discovery
  - Test-Time Adaptation
  - Prototype Learning
  - Category Explosion
  - Semantic Shift
date: 2026-05-08
content_hash: 97883c186feee3a0
---

# TALON: Test-time Adaptive Learning for On-the-Fly Category Discovery

**Conference**: CVPR2026
**arXiv**: [2603.08075](https://arxiv.org/abs/2603.08075)
**Code**: [ynanwu/TALON](https://github.com/ynanwu/TALON)
**Area**: Model Compression / Open-World Learning
**Keywords**: On-the-Fly Category Discovery, Test-Time Adaptation, Prototype Learning, Category Explosion, Semantic Shift

## TL;DR

This paper proposes TALON, the first test-time adaptive framework for On-the-Fly Category Discovery (OCD). By combining semantics-aware prototype updating, stable encoder adaptation, and margin-aware logit calibration, TALON operates directly in continuous feature space without hash encoding, substantially alleviating category explosion and significantly improving novel category discovery accuracy.

## Background & Motivation

1. **Closed-world assumption limitations**: Traditional visual recognition systems assume all categories are predefined, making them incapable of discovering new concepts or generalizing beyond the training set.
2. **OCD task definition**: On-the-Fly Category Discovery represents the most realistic open-world setting—only labeled data from known categories is available during offline training, while unlabeled data streams are processed instance-by-instance online, requiring simultaneous recognition of known categories and discovery of novel ones.
3. **Deficiencies of hash encoding**: Existing OCD methods (SMILE, PHE) freeze the feature extractor and quantize features into binary hash codes as class prototypes. Quantization causes information loss, reduced representational capacity, and amplified intra-class variance, which readily leads to **category explosion** (a single true category fragmented into multiple pseudo-categories).
4. **Static inference is unreasonable**: Fixing both the encoder and prototypes during the online phase entirely ignores the learning potential of newly arriving data—a stance at odds with the principle of "learning through discovery."
5. **Generative methods remain limited**: Although DiffGRE synthesizes novel-class samples via diffusion models, it projects features into a lower-dimensional space and remains fundamentally insufficient.
6. **Conventional TTA is ill-suited**: Existing TTA methods (TENT, MEMO, etc.) are designed for domain shift rather than semantic/label-space shift, and perform poorly or even degrade in OCD scenarios.

## Method

### Overall Architecture

TALON consists of two stages—**offline training** and **online adaptive inference**—built on a ViT-B-16 backbone (supporting DINO/CLIP pretraining), with only the last transformer block fine-tuned.

### Offline Stage: Representation Learning + Margin-aware Logit Calibration (MLC)

- **Supervised contrastive loss** $\mathcal{L}^{\text{sup}}$: pulls same-class features together and pushes apart different-class features.
- **Cross-entropy loss** $\mathcal{L}^{\text{ce}}$: a linear projection head produces logits to enhance class-level discrimination.
- **MLC module**: introduces an angular margin $m$ on the cosine similarity between normalized features and class weights:
    - Applies $s \cos(\theta_{i,y_i} + m)$ to the ground-truth class logit, while keeping $s \cos\theta_{i,c}$ for all other classes.
    - Enlarges inter-class angular distance (27.98° → 74.15°) and compresses intra-class angular distance (64.55° → 35.83°).
    - Reserves embedding space for future novel category discovery.
- Final training loss: $\mathcal{L}_{\text{labeled}} = \mathcal{L}^{\text{sup}} + \lambda \mathcal{L}^{\text{ce-m}}$

### Online Stage: Adaptive Inference

**① Online inference and novel category detection**

- A prototype memory bank $\mathcal{P}$ is maintained, with each known category initialized using the mean feature of its labeled samples.
- For each test sample, the maximum cosine similarity to all prototypes is computed; if it exceeds threshold $\tau$, the sample is assigned to a known category, otherwise a new prototype is created.

**② Semantics-aware prototype updating (TTA-P)**

- The mean feature $\bar{\mathbf{z}}_j$ and confidence $\text{conf}_j$ of samples assigned to prototype $j$ within each batch are computed.
- Adaptive step-size EMA update: $\alpha_j = \eta \cdot \text{conf}_j \cdot \frac{n_j}{n_j + \kappa}$
- High-confidence, sample-rich assignments trigger large updates; low-confidence or sample-scarce cases produce minimal updates—effectively suppressing persistent pseudo-categories caused by outliers.

**③ Stable encoder adaptation (TTA-M)**

- Small batches of test samples are collected periodically to update encoder parameters via lightweight gradient steps.
- Three losses are jointly optimized:
    - **Entropy minimization** $\mathcal{L}_{\text{ent}}$: encourages high-confidence predictions.
    - **Alignment loss** $\mathcal{L}_{\text{align}}$: keeps feature means consistent with stored prototypes.
    - **Separation loss** $\mathcal{L}_{\text{sep}}$: prevents collapse of features from different categories.
- $\mathcal{L}_{\text{TTA}} = \mathcal{L}_{\text{ent}} + \beta_1 \mathcal{L}_{\text{align}} + \beta_2 \mathcal{L}_{\text{sep}}$, with gradients propagated only through the encoder.

### Key Designs

- **Hash-free**: operates directly in continuous feature space, avoiding quantization information loss.
- **Immediate feedback + periodic update**: each instance is predicted in real time, while the model and prototypes are updated periodically, balancing responsiveness and stability.
- Known and novel categories use different update rates and smoothing constants ($\eta$=0.06/0.3, $\kappa$=32/8).

## Key Experimental Results

### Main Results (7 benchmarks, Strict-Hungarian protocol)

| Dataset | Method | All | Old | New |
|--------|------|-----|-----|-----|
| CIFAR-10 | PHE | 53.1 | 19.3 | 70.0 |
| CIFAR-10 | **TALON-DINO** | **65.0** | 46.1 | **79.3** |
| CIFAR-100 | PHE | 56.0 | 70.1 | 27.8 |
| CIFAR-100 | **TALON-DINO** | **64.7** | **77.4** | **39.3** |
| ImageNet-100 | PHE | 39.2 | 49.3 | 34.1 |
| ImageNet-100 | **TALON-DINO** | **82.6** | **92.0** | **63.4** |
| CUB-200 | DiffGRE+P | 37.9 | 57.0 | 28.3 |
| CUB-200 | **TALON-CLIP** | **45.5** | **60.7** | **37.8** |
| Stanford Cars | DiffGRE+P | 32.1 | 63.3 | 16.9 |
| Stanford Cars | **TALON-CLIP** | **53.5** | **74.2** | **43.6** |

On ImageNet-100, overall accuracy jumps from 39.2% to 82.6% (+43.4 pp), representing an exceptionally large gain.

### Category Explosion Mitigation (CUB-200 & Stanford Cars)

| Method | CUB #Cls (true: 200) | SCars #Cls (true: 196) |
|------|---------------------|----------------------|
| SMILE-64bit | 2910 | 4788 |
| PHE-64bit | 493 | 917 |
| **TALON** | **153** | **299** |

TALON's estimated category count is closest to the ground truth, effectively mitigating category explosion.

### Ablation Study (CLIP backbone, Strict-Hungarian)

| Configuration | CUB All | SCars All |
|------|---------|-----------|
| Baseline | 44.5 | 47.8 |
| +MLC | 45.7 | 49.0 |
| +MLC+TTA-P | 46.7 | 52.7 |
| +MLC+TTA-M | 46.7 | 52.1 |
| **TALON (full)** | **45.5** | **53.5** |

Each module contributes incremental gains; MLC provides better initialization, while TTA-P and TTA-M offer complementary improvements.

### Comparison with Existing TTA Methods (Stanford Cars)

| Method | All | New |
|------|-----|-----|
| Baseline+MLC+TENT | 48.1 | 39.2 |
| Baseline+MLC+OSTTA | 47.2 | 39.9 |
| **TALON** | **53.5** | **43.6** |

Conventional TTA methods are nearly ineffective or even degrade under semantic shift scenarios.

## Highlights & Insights

- **First application of TTA to OCD**: breaks the "frozen inference" paradigm and realizes "learning through discovery."
- **Hash-free framework**: operating directly in continuous feature space preserves full representational capacity and entirely avoids quantization-induced category explosion.
- **Confidence-gated adaptive prototype updating**: the dual-gating mechanism $\text{conf} \times \frac{n}{n+\kappa}$ elegantly balances update magnitude and stability.
- **Prospective embedding space shaping via MLC**: the offline stage proactively reserves space for future novel categories, with angular visualization confirming the effect (inter-class angular distance expands from 28° to 74°).
- **Hyperparameter sharing**: nearly identical configurations are used across all datasets, demonstrating strong generalizability.

## Limitations & Future Work

1. **Sensitivity to threshold $\tau$**: novel category detection relies entirely on a cosine similarity threshold that requires per-dataset tuning (0.7 for DINO, 0.75 for CLIP).
2. **Full TALON slightly underperforms +MLC+TTA-P on CUB ablation (All metric)**: suggesting that encoder adaptation (TTA-M) may introduce marginal noise on fine-grained datasets.
3. **Category count estimation remains biased**: CUB (true: 200) is underestimated at 153; SCars (true: 196) is overestimated at 299.
4. **Efficiency of instance-level online processing**: maintaining a growing prototype memory bank and periodically updating the encoder incur non-trivial computational and memory costs in long-horizon streaming scenarios.
5. **No prototype merging mechanism**: there is no explicit strategy for merging or pruning prematurely created spurious prototype entries.

## Related Work & Insights

| Dimension | SMILE/PHE (hash-based) | DiffGRE (generative) | **TALON** |
|------|---------------------|-------------------|-----------|
| Feature space | Binary hash codes | Low-dim projection | Continuous feature space |
| Online learning | ✗ Frozen | ✗ Frozen | ✓ Dual update: encoder + prototypes |
| Category explosion | Severe ($2^L$-scale inflation) | Moderate | Effectively mitigated |
| Novel class accuracy | Low | Moderate | Significantly improved |
| Additional training cost | None | Diffusion model training | Lightweight TTA |

## Rating

- Novelty: ⭐⭐⭐⭐ — First introduction of TTA into OCD; hash-free + dual-layer adaptive design is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 7 datasets, 2 backbones, 2 evaluation protocols, extensive ablations and visualizations.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear, formulations are complete, and pseudocode is well-structured.
- Value: ⭐⭐⭐⭐ — Offers substantive advances for the OCD task; the category explosion mitigation strategy is practically useful.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Learning through Creation: A Hash-Free Framework for On-the-Fly Category Discovery](learning_through_creation_a_hash-free_framework_for_on-the-fly_category_discover.md)
- [\[CVPR 2026\] FOZO: Forward-Only Zeroth-Order Prompt Optimization for Test-Time Adaptation](fozo_forward-only_zeroth-order_prompt_optimization_for_test-time_adaptation.md)
- [\[CVPR 2026\] Towards Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](iapl_aigenerated_image_detection_adaptive_prompt.md)
- [\[AAAI 2026\] Correcting False Alarms from Unseen: Adapting Graph Anomaly Detectors at Test Time](../../AAAI2026/model_compression/correcting_false_alarms_from_unseen_adapting_graph_anomaly_detectors_at_test_tim.md)
- [\[AAAI 2026\] Towards Test-time Efficient Visual Place Recognition via Asymmetric Query Processing](../../AAAI2026/model_compression/towards_test-time_efficient_visual_place_recognition_via_asymmetric_query_proces.md)

<!-- RELATED:END -->
