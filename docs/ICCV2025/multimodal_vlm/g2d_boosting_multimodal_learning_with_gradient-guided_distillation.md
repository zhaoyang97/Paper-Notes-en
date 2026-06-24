---
title: >-
  [Paper Note] G2D: Boosting Multimodal Learning with Gradient-Guided Distillation
description: >-
  [ICCV 2025][Multimodal VLM][modality imbalance] This paper proposes G2D (Gradient-Guided Distillation), which addresses the modality imbalance problem in multimodal learning by combining feature distillation and logit distillation from unimodal teachers to a multimodal student, together with a Sequential Modality Prioritization (SMP) gradient modulation strategy guided by unimodal teacher confidence scores. G2D achieves 85.89% accuracy on CREMA-D…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "modality imbalance"
  - "knowledge distillation"
  - "gradient modulation"
  - "sequential modality prioritization"
  - "multimodal fusion"
date: 2026-05-08
content_hash: fe223d5630e375bb
---

# G2D: Boosting Multimodal Learning with Gradient-Guided Distillation

**Conference**: ICCV 2025  
**arXiv**: [2506.21514](https://arxiv.org/abs/2506.21514)  
**Code**: [GitHub](https://github.com/rakib062/G2D)  
**Area**: Multimodal VLM / Modality Imbalance / Knowledge Distillation  
**Keywords**: modality imbalance, knowledge distillation, gradient modulation, sequential modality prioritization, multimodal fusion

## TL;DR
This paper proposes G2D (Gradient-Guided Distillation), which addresses the modality imbalance problem in multimodal learning by combining feature distillation and logit distillation from unimodal teachers to a multimodal student, together with a Sequential Modality Prioritization (SMP) gradient modulation strategy guided by unimodal teacher confidence scores. G2D achieves 85.89% accuracy on CREMA-D, surpassing all state-of-the-art methods focused on modality imbalance.

## Background & Motivation

**Modality Imbalance**: During joint multimodal training, one modality dominates the optimization process while others are suppressed — a phenomenon referred to as "modality competition" or "modality laziness." This leads to (i) multimodal performance falling below that of unimodal models, or (ii) degradation of weaker modality features under joint training.

**Illustrative Case on CREMA-D** (Figure 1):
   - Audio trained alone achieves 61.69%, dropping slightly to 59.95% under multimodal training.
   - **Video** trained alone achieves 76.48%, but collapses to only 27.42% under joint multimodal training.
   - The joint multimodal model achieves only 67.47%, far below the unimodal video performance of 76.48%.

**Limitations of Prior Work**:
   - **Gradient modulation** (OGM-GE, AGM): dynamically adjusts gradients for weaker modalities, but requires careful hyperparameter tuning.
   - **Feature rebalancing** (MLA, MMPareto): adjusts per-modality contributions but cannot fully eliminate imbalance.
   - **Knowledge distillation** (UMT, UME): uses unimodal teachers to guide a multimodal student, but the choice of distillation strategy requires empirical adjustment.

**Core Insight**: The fundamental issue for weak modalities is insufficient optimization — during joint training, the dominant modality converges quickly, causing gradient signals to predominantly serve it. The solution is not to "suppress the strong modality," but rather to provide the weak modality with "dedicated, undisturbed training phases."

## Method

### Overall Architecture
G2D consists of three core components: (1) independently pretrained unimodal teachers $\{T^m\}_{m=1}^k$, (2) a jointly trained multimodal student $S$, and (3) the combined G2D distillation loss $\mathcal{L}_{\text{G2D}}$ paired with the SMP gradient modulation strategy.

### Key Design 1: G2D Loss Function

Three loss terms are combined:

**(1) Multimodal student loss $\mathcal{L}_S$**: Standard cross-entropy (classification) or MSE (regression) applied to predictions from fused multimodal features.

**(2) Feature distillation loss $\mathcal{L}_{\text{feat}}$**: L2 distance aligns the student encoder's modality-specific features with those of the corresponding teacher encoder:

$$\mathcal{L}_{\text{feat}}^m = \mathbb{E}_{x \sim \mathcal{D}}\left[\|\phi_s^m(x^m; \theta_s^m) - \phi_t^m(x^m; \theta_t^m)\|^2\right]$$

**(3) Logit distillation loss $\mathcal{L}_{\text{logit}}$**: KL divergence aligns the multimodal student's output distribution with those of individual unimodal teachers:

$$\mathcal{L}_{\text{logit}}^m = \mathbb{E}_{x \sim \mathcal{D}}\left[\text{KL}(\sigma(l_t^m) \| \sigma(l_s))\right]$$

**Total G2D loss**:

$$\mathcal{L}_{\text{G2D}} = \mathcal{L}_S + \alpha \sum_{m=1}^{k} \mathcal{L}_{\text{feat}}^m + \beta \sum_{m=1}^{k} \mathcal{L}_{\text{logit}}^m$$

Feature distillation preserves modality-specific representations, while logit distillation aligns decision boundaries; the two are complementary.

### Key Design 2: Modality Confidence Scoring Module

Batch-wise average softmax probabilities from unimodal teachers serve as modality confidence scores:

$$\rho_t^m = \frac{1}{|\mathcal{B}^m|} \sum_{(x_i^m, y_i^m) \in \mathcal{B}^m} \text{Softmax}(l_t^m(x_i^m; \theta^m))[y_i^m]$$

Modalities with higher confidence are considered "dominant," while those with lower confidence are "weak." A key advantage is that confidence is measured from **unimodal** teachers, which are unaffected by the modality imbalance present in joint training.

### Key Design 3: Sequential Modality Prioritization (SMP)

**Core Hypothesis**: Providing weak modalities with dedicated, undisturbed training phases can alleviate modality imbalance.

**Procedure**:
1. Rank modalities by teacher confidence: $\pi_t[1]$ (weakest) to $\pi_t[k]$ (strongest).
2. Training proceeds in stages: the first $\tau_1$ epochs train only the weakest modality, the next $\tau_2$ epochs train the second weakest, and finally all modalities are trained jointly.
3. A gradient modulation coefficient $\kappa_q^m$ controls which modalities participate in gradient updates:

$$\theta_{q+1}^m = \theta_q^m - \eta \cdot \kappa_q^m \cdot \frac{\partial \mathcal{L}_{\text{G2D}}}{\partial \theta_q^m}$$

where $\kappa_q^m = 1$ enables training for that modality and $\kappa_q^m = 0$ freezes it.

This is a **complete suppression** strategy — rather than continuously downweighting strong modalities (as in OGM-GE via $1 - \tanh(x)$), SMP zeroes out the gradients of dominant modalities entirely, ensuring weak modalities receive fully undisturbed optimization.

## Key Experimental Results

### Datasets
- **CREMA-D**: Audio-video emotion recognition, 6 classes
- **AV-MNIST**: Audio-video digit classification, 10 classes
- **VGGSound**: Audio-video event classification, 309 classes
- **UR-Funny**: Text-visual-audio humor detection, 2 classes
- **IEMOCAP**: Audio-video-text emotion recognition
- **MIS-ME**: Soil image + meteorological tabular regression (first evaluation of modality imbalance in a regression setting)

### Main Results (Bimodal Audio-Video)

| Method | CREMA-D Multi | AV-MNIST Multi | VGGSound Multi |
|------|--------------|----------------|----------------|
| Joint-Train | 67.47 | 69.77 | 50.97 |
| AGM | 78.48 | 72.14 | 47.11 |
| OGM-GE | 58.60 | 24.53 | 37.96 |
| MLA | 79.70 | 65.32 | 51.65 |
| ReconBoost | 83.62 | 72.14 | 52.74 |
| DLMG | 67.61 | 72.33 | 53.78 |
| UMT (KD baseline) | 67.61 | 72.33 | 53.78 |
| **G2D (Ours)** | **85.89** | **73.03** | **53.82** |

G2D substantially outperforms all baselines on CREMA-D (+2.27 vs. ReconBoost), recovering video modality performance from 27.42% under joint training to 72.72%.

### Three-Modality Experiment (UR-Funny)

| Modality Combination | Joint-Train | OGM-GE | MMPareto | ReconBoost | UMT | **G2D** |
|---------|-------------|--------|----------|------------|-----|---------|
| A-V Multi | 61.57 | 61.87 | 61.27 | 62.07 | 60.46 | **62.98** |
| A-TXT Multi | 62.17 | 62.47 | 62.88 | 61.06 | 62.47 | **63.28** |
| A-V-TXT Multi | 62.58 | 63.68 | 62.88 | 61.37 | 63.38 | **65.49** |

G2D remains effective in the three-modality setting without excessively suppressing dominant modalities such as text.

### Ablation Study

| Gain of SMP Applied to Different Methods | w/o SMP | w/ SMP | Gain |
|---|---|---|---|
| Joint-Train on CREMA-D | 67.47 | 80.78 | +13.31 |
| UMT on CREMA-D | 67.61 | 82.39 | +14.78 |
| G2D loss on CREMA-D | 78.63 | 85.89 | +7.26 |

| Complete vs. Partial Suppression | CREMA-D | AV-MNIST | VGGSound | UR-Funny |
|---|---|---|---|---|
| Partial suppression (OGM-GE style) | 81.99 | 72.83 | 51.16 | 63.68 |
| **Complete suppression (SMP)** | **85.89** | **73.03** | **53.82** | **65.49** |

### Key Findings
1. SMP is effective across all methods — even applied to vanilla joint training, it yields a gain of over +13 percentage points.
2. Complete gradient suppression consistently outperforms partial suppression, supporting the hypothesis that weak modalities benefit from fully undisturbed training.
3. G2D is the first to validate the existence and mitigation of modality imbalance in a regression task (MIS-ME).
4. Late fusion performs best within the G2D framework, as it preserves independent unimodal representations.

## Highlights & Insights
1. **SMP is simple yet highly effective**: Completely freezing dominant modalities while training only the weak modality proves more effective than fine-grained gradient weight adjustment — suggesting that weak modalities need not *more* gradient, but *undisturbed* gradient.
2. **Appropriate use of knowledge distillation**: Combining feature distillation and logit distillation with supervised loss both preserves optimal unimodal representations and optimizes the multimodal objective.
3. **Strong generalizability**: G2D applies to 2- and 3-modality settings, classification and regression tasks, and multiple fusion strategies; SMP can also be used as a plug-and-play component for other methods.

## Limitations & Future Work
1. The SMP hyperparameters $\tau_j$ require tuning across datasets (e.g., the optimal setting on CREMA-D is 150 epochs for weak modality training).
2. Pretraining multiple unimodal teacher models introduces additional overall training overhead.
3. Gains are marginal on datasets with mild modality imbalance, such as AV-MNIST (73.03 vs. 72.76).
4. The approach has not been validated on large-scale pretrained models such as CLIP or LLaVA.

## Related Work & Insights
- **Gradient modulation**: OGM-GE, AGM, PMR
- **Feature rebalancing**: MLA, MMPareto, ReconBoost
- **Knowledge distillation**: UMT, UME
- **Modality imbalance analysis**: MSES, MSLR

## Rating
- **Novelty**: 3/5 (The combination of a KD framework with gradient modulation is relatively straightforward; SMP is simple but effective.)
- **Technical Depth**: 3/5 (The method is clearly presented but lacks theoretical analysis; a deeper explanation of why SMP works would strengthen the paper.)
- **Experimental Thoroughness**: 5/5 (Six datasets, 10+ baselines, extensive ablations, a regression task, and fusion strategy comparisons.)
- **Writing Quality**: 4/5 (Well-structured with rich figures and tables.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Anchor-Guided Gradient Alignment for Incomplete Multimodal Learning](../../CVPR2026/multimodal_vlm/anchor-guided_gradient_alignment_for_incomplete_multimodal_learning.md)
- [\[ICCV 2025\] A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition](a_qualityguided_mixture_of_scorefusion_experts_framework_for.md)
- [\[ICCV 2025\] BASIC: Boosting Visual Alignment with Intrinsic Refined Embeddings in Multimodal Large Language Models](basic_boosting_visual_alignment_with_intrinsic_refined_embeddings_in_multimodal_.md)
- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](controlling_multimodal_llms_via_rewardguided_decoding.md)
- [\[NeurIPS 2025\] Multimodal Negative Learning](../../NeurIPS2025/multimodal_vlm/multimodal_negative_learning.md)

</div>

<!-- RELATED:END -->
