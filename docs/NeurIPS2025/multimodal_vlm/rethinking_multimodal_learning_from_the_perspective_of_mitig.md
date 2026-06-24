---
title: >-
  [Paper Note] Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion
description: >-
  [NeurIPS 2025][Multimodal VLM][modality imbalance] This paper proposes a **classification ability disproportion** perspective to understand modality imbalance in multimodal learning, and designs a Sustained Boosting algorithm (shared encoder + multiple configurable classifiers, jointly optimizing classification and residual errors) coupled with Adaptive Classifier Assignment (ACA). The paper theoretically proves that the cross-modal gap loss converges at $\mathcal{O}(1/T)$…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "modality imbalance"
  - "classification ability disproportion"
  - "sustained boosting"
  - "adaptive classifier assignment"
date: 2026-05-08
content_hash: c9810bae68b02f41
---

# Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion

**Conference**: NeurIPS 2025
**arXiv**: [2502.20120](https://arxiv.org/abs/2502.20120)  
**Code**: [https://github.com/njustkmg/NeurIPS25-AUG](https://github.com/njustkmg/NeurIPS25-AUG)  
**Authors**: Qing-Yuan Jiang, Longfei Huang, Yang Yang (Nanjing University of Science and Technology / Nanjing University)
**Area**: Multimodal Learning · Modality Imbalance
**Keywords**: modality imbalance, classification ability disproportion, sustained boosting, adaptive classifier assignment

## TL;DR

This paper proposes a **classification ability disproportion** perspective to understand modality imbalance in multimodal learning, and designs a Sustained Boosting algorithm (shared encoder + multiple configurable classifiers, jointly optimizing classification and residual errors) coupled with Adaptive Classifier Assignment (ACA). The paper theoretically proves that the cross-modal gap loss converges at $\mathcal{O}(1/T)$, and achieves substantial improvements over SOTA on six datasets including CREMAD.

## Background & Motivation

The core bottleneck of multimodal learning (MML) is **modality imbalance**: different modalities converge at significantly different rates during joint training. On the CREMAD dataset, the audio modality (dominant) achieves ~63% unimodal accuracy, while video (weak) reaches only ~45%—a substantial gap. Existing solutions fall into two categories:

1. **Regulating the learning process**—OGM performs gradient modulation, MSLR adjusts learning rates, G-Blend adapts fusion weights; all essentially slow down the dominant modality or accelerate the weaker one.
2. **Enhancing modality interaction**—MLA alternates training to transfer optimization information, ReconBoost uses gradient boosting to capture cross-modal complementary information, and DI-MML injects cross-modal optimization signals.

**Key Insight**: These methods all operate at the level of "balancing learning speed," overlooking a more fundamental issue—the **classification capability of the weak modality's classifier is itself insufficient**. Even if learning speeds are balanced, insufficient classifier capacity for the weak modality cannot match the dominant one.

The authors validate this with a toy experiment: applying additional gradient boosting to the video modality (with audio unchanged) after naive MML training raises video accuracy from ~45% to ~65%+, and overall accuracy from 0.6507 to a level far exceeding G-Blend. This demonstrates that **directly enhancing the weak modality's classification ability** is both feasible and effective.

## Core Problem

How to **directly improve the classification capability of the weak modality** within a joint multimodal training framework, so that the classification performance of dominant and weak modalities tends toward equilibrium—rather than merely balancing learning speeds?

## Method

### Overall Architecture

Each modality employs: a **shared encoder** $\phi^o(\cdot)$ to extract features $\boldsymbol{u}^o$, plus **multiple configurable classifiers** $\psi_t^o(\cdot)$ to produce predictions. The encoder shares parameters across classifiers, and the final layer (Layer2) of each classifier is shared across modalities to enhance interaction.

### 1. Sustained Boosting Algorithm

Inspired by gradient boosting, $n$ classifiers are trained for the weak modality to progressively learn the residual of preceding classifiers. The **residual label** for the $t$-th classifier is:

$$\hat{\boldsymbol{y}}_{it}^o = \boldsymbol{y}_i - \lambda \sum_{j=1}^{t-1} \boldsymbol{y}_i \odot \boldsymbol{p}_{ij}^o$$

where $\lambda \in [0,1]$ controls label smoothing, $\odot$ denotes element-wise multiplication, and the $\boldsymbol{y}_i$ mask ensures non-negative residuals.

The total loss consists of three terms:
- **Residual error** $\epsilon$: cross-entropy of the $t$-th classifier against the residual label—learning new information.
- **Overall error** $\epsilon_{\text{all}}$: cross-entropy of the sum of all $t$ classifiers' predictions against the ground truth—ensuring overall accuracy.
- **Maintenance error** $\epsilon_{\text{pre}}$: cross-entropy of the sum of the first $t\!-\!1$ classifiers against the ground truth—preventing degradation of existing classifiers due to shared encoder updates.

$$L(\boldsymbol{x}_i^o, \boldsymbol{y}_i, t) = \epsilon + \epsilon_{\text{all}} + \epsilon_{\text{pre}}$$

**Key distinction from traditional gradient boosting**: Conventional methods are stage-wise (freezing previous stages), whereas this work **simultaneously and continuously optimizes** all classifiers and the encoder—hence the term "sustained" boosting.

### 2. Adaptive Classifier Assignment (ACA)

Since the inter-modality gap changes dynamically during training, static classifier allocation is insufficiently flexible. ACA checks every $t_N$ epochs using a confident score:

$$s_t^o = \frac{1}{N}\sum_{i=1}^{N} \boldsymbol{y}_i^\top \left[\sum_{j=1}^{n^o} \boldsymbol{p}_{ij}^o\right]$$

If $s_t^a - \sigma \cdot s_t^v > \tau$ (audio significantly outperforms video), a **new classifier is added for video**; the converse likewise. Default values are $\sigma=1.0$ and $\tau=0.01$.

### 3. Configurable Classifier Architecture

Each classifier is a lightweight two-layer fully connected network: Layer1($D \times 256$) → ReLU → Layer2($256 \times K$). Adding one classifier introduces only ~1M parameters (compared to ResNet18's 11.8M), incurring minimal cost.

### 4. Theoretical Guarantee

Defining the cross-modal gap function $\mathcal{G}(\Phi) = \mathcal{L}^a(\Phi^a) - \mathcal{L}^v(\Phi^v)$, under standard assumptions including Lipschitz smoothness:

$$\mathcal{G}(\Phi(T)) \leq \frac{\mathcal{G}(\Phi(0))}{1 + \frac{\nu^2\kappa^2}{2L_a\beta^2} T \cdot \mathcal{G}(\Phi(0))}$$

That is, the gap loss converges at $\mathcal{O}(1/T)$—the loss of the weak modality will progressively catch up to that of the dominant modality.

## Experimental Setup

**Datasets**: Six multimodal datasets covering audio-visual tasks (CREMAD, KSounds, VGGSound), trimodal gesture recognition (NVGesture: RGB+OF+Depth), and image-text tasks (Twitter, Sarcasm).

**Baselines**: Standard fusion methods (Concat, Affine, ML-LSTM, Sum, Weight) and rebalancing methods (MSES, G-Blend, MSLR, OGM, PMR, AGM, MMPareto, SMV, MLA, DI-MML, LFM, ReconBoost)—17 comparison methods in total.

**Implementation Details**:
- Encoders: ResNet18 (audio-visual), I3D (NVGesture), BERT + ResNet50 (image-text)
- Optimizer: SGD, lr=0.01, momentum=0.9, weight decay=1e-4; Adam with lr=2e-5 for image-text
- $\lambda$ searched from {0.1, 0.2, 0.33, 0.5, 1.0}
- $t_N$ (check interval): 20 epochs for CREMAD; 10 for VGGSound/KSounds/NVGesture; 5 for Twitter; 1 for Sarcasm

## Key Experimental Results

### Main Results

| Dataset | Modalities | Naive MML | Best Baseline | **Ours** |
|--------|------|-----------|--------------|---------|
| CREMAD | Audio+Video | 0.6507 | 0.8362 (LFM) | **0.8515** |
| KSounds | Audio+Video | 0.6455 | 0.7253 (LFM) | **0.7263** |
| VGGSound | Audio+Video | 0.5116 | 0.5274 (LFM) | **0.5301** |
| Twitter | Image+Text | 0.7300 | 0.7501 (LFM) | **0.7512** |
| Sarcasm | Image+Text | 0.8294 | 0.8497 (LFM) | **0.8510** |
| NVGesture | RGB+OF+Depth | 0.8237 | 0.8436 (LFM) | **0.8501** |

The proposed method achieves the best results on all datasets. On CREMAD, it improves by **20 percentage points** over Naive MML and by 1.5% over the previous best, LFM.

### Ablation Study (CREMAD)

| Loss Combination | Multimodal Acc | Audio Acc | Video Acc |
|---------|-----------|---------|---------|
| $\epsilon$ only (residual) | 0.8333 | 0.6465 | 0.6734 |
| $\epsilon_{\text{all}}$ only (overall) | 0.8320 | 0.6573 | 0.6707 |
| $\epsilon_{\text{pre}}$ only (maintenance) | 0.8360 | 0.6841 | 0.6371 |
| **All three combined** | **0.8515** | 0.6835 | **0.6828** |

All three loss terms are indispensable; joint optimization simultaneously achieves the highest multimodal accuracy and the most balanced per-modality performance.

### Adaptive vs. Fixed Classifier Allocation

| Strategy | Audio/Video Classifiers | Multimodal Acc |
|------|----------------|-----------|
| Fixed 10 video classifiers | 1+10 | 0.8091 |
| Fixed 12 video classifiers | 1+12 | 0.8118 |
| **Adaptive (ACA)** | 1+10 (dynamic) | **0.8515** |

With the same final number of classifiers, adaptive allocation outperforms fixed allocation by **4.2%**, indicating that *when* classifiers are added matters more than *how many* are added.

### Model Capacity Control

| Method | Architecture | Parameters | Acc |
|------|------|-------|-----|
| Naive MML | R18+R18 | 23.6M | 0.6507 |
| Naive MML | R18+R34 | 35.1M | 0.6277 |
| **Ours** | R18+R18+classifiers | **24.6M** | **0.8515** |

Scaling from R18 to R34 adds 11.5M parameters yet degrades accuracy (harder to converge), whereas the proposed method adds only 1M parameters and achieves a 20% improvement—demonstrating that performance gains stem from the boosting mechanism, not parameter count.

### Robustness to Missing Modalities (CREMAD)

| Method | 0% Missing | 20% Missing | 50% Missing |
|------|--------|---------|---------|
| Naive MML | 0.6507 | 0.5849 | 0.5242 |
| MLA | 0.7943 | 0.6935 | 0.5753 |
| **Ours** | **0.8515** | **0.7540** | **0.6008** |

At 50% missing modality, the proposed method (0.6008) still surpasses MLA on complete data (0.5753 @ 50% missing). t-SNE visualizations further show that the proposed method learns significantly more discriminative features for the video modality compared to naive MML and ReconBoost.

## Highlights & Insights

- **Novel Perspective**: Framing the problem as *classification ability disproportion* rather than *learning speed imbalance* addresses a more fundamental cause of modality imbalance.
- **Elegant Adaptation of Boosting**: Residual learning from gradient boosting is seamlessly embedded into joint multimodal training; sustained simultaneous optimization avoids the information loss inherent to stage-wise methods.
- **Theoretical Completeness**: Proving $\mathcal{O}(1/T)$ convergence of the gap loss provides rigorous theoretical grounding for the method's effectiveness.
- **Rigorous Experimental Design**: Model capacity control experiments rule out the alternative explanation of "more parameters," and the adaptive vs. fixed comparison establishes the necessity of dynamic allocation.

## Limitations & Future Work

- Theoretical analysis covers only the effect of boosting on gap convergence; overall convergence of the complete framework is not proved.
- The number of classifiers grows monotonically during training, introducing linear inference overhead.
- Validation is limited to classification tasks; downstream tasks such as retrieval, generation, and segmentation remain unexplored.
- Adaptation to early fusion architectures is not discussed.
- The hyperparameter $t_N$ is set manually per dataset, lacking an adaptive selection mechanism.

## Related Work & Insights

- **vs. OGM**: OGM slows dominant modality learning via gradient modulation; this work directly increases weak modality classifier capacity—addressing a more fundamental issue.
- **vs. MLA**: MLA alternates training to transfer optimization information and bridge the gap; this work explicitly enhances the weak modality via boosting—achieving 1.5% higher accuracy on CREMAD (vs. LFM).
- **vs. ReconBoost**: Both employ gradient boosting, but ReconBoost aims to iteratively capture cross-modal complementary information, whereas this work directly improves weak modality classification ability—differing in both motivation and mechanism.
- **vs. Increasing Network Capacity**: Scaling R18→R34 (+11.5M parameters) yields worse results; the multi-classifier ensemble (+1M parameters) performs far better—structure matters more than capacity.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The classification ability disproportion perspective and the application of sustained boosting in MML offer clear originality.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six datasets, 17 baselines, ablation studies, strategy comparisons, capacity control, missing modality robustness, and t-SNE visualization.
- **Writing Quality**: ⭐⭐⭐⭐ The toy experiment provides clear intuition; the problem→method→theory logical chain is complete.
- **Value**: ⭐⭐⭐⭐ Provides a new theoretical framework and practical approach for multimodal imbalance; the 20% absolute improvement on CREMAD is remarkable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Multimodal Negative Learning](multimodal_negative_learning.md)
- [\[CVPR 2025\] GeoMM: On Geodesic Perspective for Multi-Modal Learning](../../CVPR2025/multimodal_vlm/geomm_on_geodesic_perspective_for_multi-modal_learning.md)
- [\[CVPR 2026\] Rethinking Cross-Modal Anchor Alignment for Mitigating Error Accumulation](../../CVPR2026/multimodal_vlm/rethinking_cross-modal_anchor_alignment_for_mitigating_error_accumulation.md)
- [\[NeurIPS 2025\] MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning](midas_misalignment-based_data_augmentation_strategy_for_imbalanced_multimodal_le.md)
- [\[NeurIPS 2025\] NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints](navil_rethinking_scaling_properties_of_native_multimodal_large_language_models_u.md)

</div>

<!-- RELATED:END -->
