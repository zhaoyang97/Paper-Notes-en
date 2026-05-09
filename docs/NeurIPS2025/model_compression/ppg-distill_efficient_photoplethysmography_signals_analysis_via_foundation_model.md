---
title: >-
  [Paper Note] PPG-Distill: Efficient Photoplethysmography Signals Analysis via Foundation Model Distillation
description: >-
  [NeurIPS 2025][Model Compression][Knowledge Distillation] PPG-Distill proposes a knowledge distillation framework tailored for PPG signals. By combining prediction-level, feature-level, and patch-level (morphology + rhythm) distillation, it transfers knowledge from large PPG foundation models to lightweight student models, achieving up to 21.8% performance improvement alongside 7× inference speedup and 19× memory compression.
tags:
  - NeurIPS 2025
  - Model Compression
  - Knowledge Distillation
  - PPG Signals
  - Foundation Model Compression
  - Waveform Morphology Distillation
  - Rhythm Distillation
date: 2026-05-08
content_hash: 034c7b9331a13dbd
---

# PPG-Distill: Efficient Photoplethysmography Signals Analysis via Foundation Model Distillation

**Conference**: NeurIPS 2025
**arXiv**: [2509.19215](https://arxiv.org/abs/2509.19215)
**Code**: [GitHub](https://github.com/LingFengGold/PPG-Distill)
**Area**: Model Compression / Knowledge Distillation / Health Signal Processing
**Keywords**: Knowledge Distillation, PPG Signals, Foundation Model Compression, Waveform Morphology Distillation, Rhythm Distillation

## TL;DR

PPG-Distill proposes a knowledge distillation framework tailored for PPG signals. By combining prediction-level, feature-level, and patch-level (morphology + rhythm) distillation, it transfers knowledge from large PPG foundation models to lightweight student models, achieving up to 21.8% performance improvement alongside 7× inference speedup and 19× memory compression.

## Background & Motivation

Photoplethysmography (PPG) is among the most widely used physiological signals in wearable health monitoring, capturing blood volume changes non-invasively. The analytical value of PPG signals derives from two complementary aspects:

**Local Waveform Morphology**: Short-window waveform characteristics that reflect cardiovascular events.

**Long-range Structural Rhythm**: Cross-patch temporal structures that encode heartbeat periodicity and autonomic nervous system regulation.

Recent PPG foundation models such as GPT-PPG and PaPaGei have demonstrated strong generalization capabilities; however, their substantial parameter counts and computational demands make deployment on edge devices such as smartwatches highly challenging.

**Key Problem**: Conventional knowledge distillation methods (e.g., output alignment, feature alignment) focus exclusively on global information, **neglecting the PPG-specific local morphological patterns and cross-patch rhythmic structures**. Although existing PPG foundation models already employ patch-based representations, these local dynamics remain underutilized during distillation.

## Method

### Overall Architecture

Building upon standard Global KD (prediction-level + feature-level distillation), PPG-Distill introduces two additional patch-level distillation strategies, forming a four-tier distillation framework:

```
PPG Signal X ∈ ℝ^L
  ├── Patchify → X_p ∈ ℝ^{P×N}  (P=patch length, N=number of patches, N=L/P)
  │
  ├── Global KD
  │   ├── Prediction-level distillation: L_KD^Y (align teacher-student outputs)
  │   └── Feature-level distillation: L_KD^H (align signal-level features)
  │
  └── Patch-Level KD (core of PPG-Distill)
      ├── Morphology distillation L_mor (intra-patch local waveform alignment)
      └── Rhythm distillation L_rhy (inter-patch temporal structure alignment)
```

### Key Designs

#### 1. PPG Morphology Distillation

**Objective**: Align the local waveform representations of teacher and student at each patch position.

- Teacher patch features: $H_t^p \in \mathbb{R}^{N \times d_t}$
- Student patch features: $H_s^p \in \mathbb{R}^{N \times d_s}$
- A learnable linear adapter $A \in \mathbb{R}^{d_t \times d_s}$ is introduced to bridge the dimensionality gap: $\tilde{H}_t^p = H_t^p A$
- After $\ell_2$ normalization of patch vectors, a similarity matrix is constructed as $Z = \frac{\hat{H}_s^p (\hat{H}_t^p)^\top}{\tau}$

An **InfoNCE contrastive loss** is adopted, where each student patch forms a positive pair with the teacher patch at the same position, and all remaining patches serve as negatives:

$$\mathcal{L}_{mor} = \frac{1}{N} \sum_{i=1}^{N} \left( -\log \frac{\exp(Z_{ii})}{\sum_{j=1}^N \exp(Z_{ij})} \right)$$

This objective encourages the student to preserve the discriminative morphological features captured by the teacher at each patch.

#### 2. PPG Rhythm Distillation

**Objective**: Preserve the inter-patch temporal relational structure (heartbeat periodicity and beat regularity).

Unlike morphology distillation, which targets individual patch alignment, rhythm distillation transfers the teacher's **inter-patch relational matrix**. Pairwise Euclidean distance matrices are constructed for both teacher and student:

$$[D_t]_{ij} = \| \phi(H_{t,i}^p) - \phi(H_{t,j}^p) \|_2, \quad [D_s]_{ij} = \| H_{s,i}^p - H_{s,j}^p \|_2$$

After normalization, a Smooth L1 loss is applied for alignment:

$$\mathcal{L}_{rhy} = \frac{1}{N(N-1)} \sum_{i \neq j} \text{smoothL1}([\tilde{D}_s]_{ij}, [\tilde{D}_t]_{ij})$$

This loss, inspired by relational knowledge distillation (Park et al., 2019), penalizes differences in relative distances rather than absolute values, making it robust to feature space scaling.

### Loss & Training

**Joint optimization objective**:

$$\mathcal{L} = \mathcal{L}_{sup} + \alpha \mathcal{L}_{KD}^Y + \beta \mathcal{L}_{KD}^H + \gamma (\mathcal{L}_{mor} + \mathcal{L}_{rhy})$$

- $\mathcal{L}_{sup}$: Supervised loss (MAE for regression, cross-entropy for classification)
- $\alpha, \beta, \gamma$: Hyperparameters controlling the weight of each loss term

**Training details**:
- The teacher model is frozen; only the student is trained
- Patch size: P = 40
- Optimizer: Adam, initial learning rate 1e-5, maximum 1e-3
- Warmup + Cosine Annealing schedule (warmup ratio 25%)
- Early stopping: patience = 20 epochs
- Temperature $\tau = 2$
- $\alpha, \beta, \gamma$ searched within {0.1, 0.5}

## Key Experimental Results

### Main Results: DaLiA Heart Rate Estimation + StanfordAF Atrial Fibrillation Detection

| Dataset | Teacher | Student | Method | Metric 1 | Metric 2 |
|:---|:---|:---|:---|:---|:---|
| **DaLiA** | GPT-PPG-19m | Teacher itself | — | MSE: 221.78 | MAE: 8.82 |
| | | GPT-PPG-1m | No distillation | MSE: 255.07 | MAE: 10.08 |
| | | GPT-PPG-1m | +Global KD | MSE: 234.16 (+8.2%) | MAE: 9.44 (+6.4%) |
| | | GPT-PPG-1m | **+PPG-Distill** | **MSE: 215.36 (+15.6%)** | **MAE: 8.34 (+17.3%)** |
| **DaLiA** | PaPaGei | Teacher itself | — | MSE: 160.39 | MAE: 6.81 |
| | | GPT-PPG-1m | No distillation | MSE: 255.07 | MAE: 10.08 |
| | | GPT-PPG-1m | +Global KD | MSE: 220.26 (+13.7%) | MAE: 8.38 (+16.9%) |
| | | GPT-PPG-1m | **+PPG-Distill** | **MSE: 202.31 (+20.7%)** | **MAE: 7.90 (+21.6%)** |
| **StanfordAF** | GPT-PPG-19m | Teacher itself | — | Acc: 0.93 | F1: 0.88 |
| | | GPT-PPG-1m | No distillation | Acc: 0.81 | F1: 0.64 |
| | | GPT-PPG-1m | +Global KD | Acc: 0.82 (+0.8%) | F1: 0.65 (+2.7%) |
| | | GPT-PPG-1m | **+PPG-Distill** | **Acc: 0.87 (+6.7%)** | **F1: 0.77 (+21.8%)** |
| **StanfordAF** | PaPaGei | Teacher itself | — | Acc: 0.83 | F1: 0.70 |
| | | GPT-PPG-1m | +Global KD | Acc: 0.83 (+1.8%) | F1: 0.67 (+5.7%) |
| | | GPT-PPG-1m | **+PPG-Distill** | **Acc: 0.88 (+7.7%)** | **F1: 0.77 (+21.4%)** |

**Key finding**: On DaLiA with GPT-PPG-19m as the teacher, the PPG-Distill-trained GPT-PPG-1m (MSE: 215.36) **surpasses the teacher** (MSE: 221.78) using only 1/19 of its parameters.

### Ablation Study: Efficiency Comparison

| Model | DaLiA MAE | Batch/s | Parameters | Memory (MB) |
|:---|:---:|:---:|:---:|:---:|
| GPT-PPG-19m | 8.82 | 128.06 | 19,018,417 | 72.6 |
| PaPaGei | 6.81 | 225.80 | 5,917,197 | 22.6 |
| MLP | 10.74 | 4248.70 | 41,473 | 0.16 |
| **GPT-PPG-1m (PPG-Distill)** | **7.90** | **291.50** | **1,017,197** | **3.9** |

| Model | StanfordAF F1 | Batch/s | Parameters | Memory (MB) |
|:---|:---:|:---:|:---:|:---:|
| GPT-PPG-19m | 0.88 | 39.19 | 19,034,290 | 72.7 |
| PaPaGei | 0.70 | 222.30 | 5,917,454 | 22.6 |
| MLP | 0.54 | 1546.70 | 154,242 | 0.59 |
| **GPT-PPG-1m (PPG-Distill)** | **0.77** | **290.00** | **1,021,690** | **3.9** |

- Inference speedup: approximately **7×** relative to GPT-PPG-19m
- Memory reduction: approximately **19×** relative to GPT-PPG-19m (72.6 MB → 3.9 MB)
- Parameter reduction: 19M → 1M (approximately **19×**)

### Key Findings

1. **PPG-Distill consistently outperforms Global KD**: Patch-level distillation yields additional gains across all task and teacher combinations.
2. **Students can surpass their teachers**: On DaLiA, the 1M-parameter student exceeds the 19M-parameter teacher, demonstrating that structured distillation can compensate for—and even reverse—capacity gaps.
3. **Teacher quality affects distillation outcomes**: In regression tasks, a stronger teacher (PaPaGei) produces a better student; this trend is less pronounced in classification tasks.
4. **MLP architectures are fundamentally limited**: Even with Global KD, MLP fails to surpass GPT-PPG-1m, indicating that shallow architectures struggle to model complex PPG dynamics.
5. **Hyperparameter sensitivity**: $\alpha$ (prediction-level weight) has the largest impact on performance; $\beta$ (feature-level) is the most stable; $\gamma$ (patch-level) exhibits an optimal value at $\gamma=1$.

## Highlights & Insights

1. **Signal-property-driven distillation design**: Rather than applying a generic distillation method, PPG-Distill designs task-specific distillation objectives grounded in the two core properties of PPG signals—morphology and rhythm.
2. **Full exploitation of existing patch structure**: Existing PPG foundation models already employ patchification, yet this intermediate representation is underutilized during distillation—PPG-Distill addresses this gap directly.
3. **Complementary use of contrastive learning and relational distillation**: Morphology distillation applies InfoNCE to preserve per-patch discriminability, while rhythm distillation employs a relational distance matrix to maintain cross-patch structural consistency; the two objectives are mutually complementary.
4. **Practical deployability**: The 19× memory compression and 7× speedup make deployment on smartwatches a realistic prospect.

## Limitations & Future Work

1. **Limited task coverage**: Validation is restricted to heart rate estimation and atrial fibrillation detection; PPG signals are also applicable to blood pressure estimation, SpO₂ monitoring, stress assessment, and other tasks.
2. **Single student architecture**: Only GPT-PPG-1m and MLP are evaluated as student models; alternative lightweight architectures (e.g., MobileNet variants, TinyTransformer) remain unexplored.
3. **Fixed patch size**: P = 40 follows the default setting of GPT-PPG; the effect of varying patch sizes on distillation performance is not analyzed.
4. **Absence of real-device deployment validation**: Efficiency analysis is conducted on GPU; performance on actual wearable hardware (ARM chips, low-power MCUs) is not evaluated.
5. **Limited teacher model diversity**: Only two PPG foundation models are tested; the effect of non-foundation-model teachers is not explored.
6. **Cross-domain generalization unverified**: Training and test data share the same distribution; the robustness of the distilled model across different devices or populations is not validated.

## Related Work & Insights

- **GPT-PPG (Chen et al., 2025)**: A generative Transformer for PPG pre-training; one of the teacher models in this work.
- **PaPaGei (Pillai et al., 2025)**: A PPG foundation model based on morphology-aware contrastive learning, pre-trained on 57,000 hours of clinical PPG data.
- **TimeDistill (Ni et al., 2025)**: Cross-architecture distillation for time-series data focusing on multi-scale and multi-periodicity features—PPG-Distill is more specialized, targeting PPG-specific morphology and rhythm.
- **Relational KD (Park et al., 2019)**: Relational knowledge distillation; PPG-Distill's rhythm distillation borrows the distance matrix alignment idea from this work.
- **InfoNCE (van den Oord et al., 2018)**: Contrastive learning loss used in PPG-Distill's morphology distillation.

The key insight of this paper is that **domain-specific signal properties should guide the design of distillation methods**. Generic distillation approaches overlook the waveform morphology and rhythmic structure of PPG signals, whereas leveraging such domain knowledge substantially improves distillation effectiveness. This principle is generalizable to foundation model compression for other physiological signals such as ECG and EEG.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First work to specialize knowledge distillation for PPG signals; the morphology + rhythm distillation design is creative.
- **Theoretical Depth**: ⭐⭐⭐ — The method is intuitive and effective, but theoretical analysis is limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-teacher, multi-task evaluation with comprehensive efficiency analysis and ablation studies.
- **Practical Value**: ⭐⭐⭐⭐⭐ — 19× memory compression and 7× speedup make edge deployment feasible; open-source code is provided.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated; method description is thorough; experimental presentation is well-organized.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Towards Effective Federated Graph Foundation Model via Mitigating Knowledge Entanglement](towards_effective_federated_graph_foundation_model_via_mitigating_knowledge_enta.md)
- [\[NeurIPS 2025\] A Token is Worth over 1,000 Tokens: Efficient Knowledge Distillation through Low-Rank Clone](a_token_is_worth_over_1000_tokens_efficient_knowledge_distillation_through_low-r.md)
- [\[NeurIPS 2025\] Accurate and Efficient Low-Rank Model Merging in Core Space](accurate_and_efficient_low-rank_model_merging_in_core_space.md)
- [\[NeurIPS 2025\] Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts](toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)
- [\[NeurIPS 2025\] Revisiting Semi-Supervised Learning in the Era of Foundation Models](revisiting_semi-supervised_learning_in_the_era_of_foundation_models.md)

<!-- RELATED:END -->
