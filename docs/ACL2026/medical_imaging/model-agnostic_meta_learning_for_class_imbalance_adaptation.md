---
title: >-
  [Paper Note] Model-Agnostic Meta Learning for Class Imbalance Adaptation
description: >-
  [ACL 2026][Medical Imaging][class imbalance] This paper proposes HAMR (Hardness-Aware Meta-Resample), a unified meta-learning framework that dynamically estimates instance-level importance weights via bi-level optimizati…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "class imbalance"
  - "meta-learning"
  - "adaptive weighting"
  - "hardness-aware resampling"
  - "bi-level optimization"
date: 2026-05-08
content_hash: 9db329cb992a1abf
---

# Model-Agnostic Meta Learning for Class Imbalance Adaptation

**Conference**: ACL 2026
**arXiv**: [2604.18759](https://arxiv.org/abs/2604.18759)
**Code**: [GitHub](https://github.com/trust-nlp/ImbalanceLearning)
**Area**: Medical Imaging
**Keywords**: class imbalance, meta-learning, adaptive weighting, hardness-aware resampling, bi-level optimization

## TL;DR

This paper proposes HAMR (Hardness-Aware Meta-Resample), a unified meta-learning framework that dynamically estimates instance-level importance weights via bi-level optimization to prioritize genuinely difficult samples, coupled with a neighborhood-aware resampling mechanism that shifts training focus toward hard samples and their semantic neighbors. HAMR consistently outperforms strong baselines across 6 imbalanced NLP datasets.

## Background & Motivation

**Background**: Class imbalance is pervasive in NLP tasks such as text classification and named entity recognition. Existing approaches fall into two main categories: loss re-weighting (e.g., Focal Loss, Dice Loss) and data resampling (oversampling / synthetic generation).

**Limitations of Prior Work**: (1) These methods typically rely on predefined static heuristics that apply uniform adjustment ratios to all samples within the same class. (2) Sample difficulty is not equivalent to class membership — not all minority-class instances are inherently hard, nor are all majority-class samples trivial. (3) Static schemes may incorrectly downweight informative majority-class samples while over-emphasizing easy minority-class instances.

**Key Challenge**: A method is needed that can dynamically identify and prioritize genuinely difficult samples — regardless of class membership — and adapt its learning strategy in accordance with the model's evolving understanding of the data.

**Goal**: To design a unified framework that simultaneously addresses class imbalance and instance-level difficulty, dynamically steering the model's learning focus.

**Key Insight**: Decouple "what the model should attend to" (adaptive weighting) from "what the model should be exposed to" (resampling) into two complementary modules, jointly optimized within a meta-learning framework.

**Core Idea**: Employ bi-level meta-optimization to dynamically learn instance importance weights — the inner loop performs an intermediate update using pre-meta weights, while the outer loop updates the weight network on a balanced meta-validation set to obtain post-meta weights for the actual model update — combined with FAISS-accelerated neighborhood-enhanced resampling to shift the training distribution toward challenging semantic regions.

## Method

### Overall Architecture

HAMR comprises two core modules: (1) **Adaptive Weight Estimation** — a lightweight weight network $f_\theta$ maps normalized per-sample losses to importance weights, dynamically adjusted via bi-level meta-optimization; and (2) **Hardness-Aware Region Resampling** — dynamically adjusts the training distribution based on EMA-smoothed difficulty scores and KNN neighborhood augmentation. Both modules operate collaboratively within a unified training loop.

### Key Designs

1. **Adaptive Weight Estimation via Bi-Level Meta-Optimization**:

    - **Function**: Dynamically adjusts the importance of each training sample based on the model's current learning state.
    - **Mechanism**: In the inner loop, the current weight network produces pre-meta weights $w_i^{\text{pre}}$ and performs an intermediate gradient update to obtain $\phi'$. In the outer loop, $f_{\phi'}$ is evaluated on a balanced meta-validation set $\mathcal{D}_{\text{meta}}$ to update the weight network parameters $\theta$. The updated weight network then recomputes post-meta weights $w_i^{\text{post}}$ for the actual model update. For token-level tasks, the sentence-level maximum token loss is used; for classification tasks, per-sample cross-entropy is used.
    - **Design Motivation**: Pre-meta weights reflect "what the model deems important before updating," while post-meta weights reflect "what is truly important under the guidance of the balanced validation set." This try-then-revise strategy adapts more effectively to training dynamics than static heuristics.

2. **Hardness-Aware Region Resampling with Neighborhood Augmentation**:

    - **Function**: Dynamically adjusts the training distribution to expose the model more frequently to challenging semantic regions.
    - **Mechanism**: EMA smoothing is applied to post-meta weights to obtain global difficulty scores: $h_i \leftarrow \gamma \cdot h_i + (1-\gamma) \cdot w_i^{\text{post}}$. The top 20% hardest samples are selected, and FAISS-accelerated KNN retrieves $k$ semantic neighbors for each hard sample, yielding a neighborhood-augmented score $b_i$. The final sampling probability is $p_i \propto (h_i + \varepsilon)^\tau \cdot (1 + \lambda b_i)$, where temperature $\tau < 1$ encourages balanced exploration.
    - **Design Motivation**: Focusing solely on isolated hard samples is insufficient — the semantic neighbors of hard samples tend to exhibit similarly challenging characteristics. Neighborhood augmentation propagates difficulty signals from individual samples to entire semantic regions.

3. **Unified Training Loop**:

    - **Function**: Seamlessly integrates weight estimation and resampling into an end-to-end training procedure.
    - **Mechanism**: Neighborhood augmentation is updated every $F$ epochs to avoid per-step KNN overhead. Each mini-batch undergoes the complete pipeline: sampling → pre-meta weights → inner update → meta step → post-meta weights → outer update → EMA update.
    - **Design Motivation**: The two modules are mutually reinforcing — weight estimation provides instance-level importance signals, while resampling ensures the model is sufficiently exposed to hard-region samples.

### Loss & Training

The primary loss is weighted cross-entropy / token-level loss. The meta-validation set is constructed by taking the full validation set and supplementing it with training samples drawn to match the median class size, forming a balanced set. Weights undergo batch-wise z-score normalization before being processed by the weight network; outputs are clipped to a fixed range to ensure numerical stability.

## Key Experimental Results

### Main Results

| Dataset | Task | HAMR Macro-F1 | Best Baseline Macro-F1 | Gain |
|---------|------|--------------|----------------------|------|
| BioNLP | NER | 72.7 | 70.6 (Dice) | +2.1 |
| TweetNER | NER | 60.2 | 59.0 (Dice/LNR) | +1.2 |
| MIT-Restaurant | NER | 81.1 | 80.4 (Dice) | +0.7 |
| Hurricane-Irma17 | CLS | 73.4 | 72.7 (ICF) | +0.7 |
| Cyclone-Idai19 | CLS | 65.7 | 63.8 (ICF) | +1.9 |
| SST-5 | CLS | 57.0 | 56.3 (ICF) | +0.7 |

### Ablation Study

| Configuration | BioNLP F1 | Irma17 F1 | Note |
|--------------|----------|----------|------|
| HAMR (full) | 72.7 | 73.4 | Complete model |
| w/o resampling | 71.4 | 72.1 | Remove neighborhood resampling |
| w/o meta-weights | 70.9 | 71.8 | Remove adaptive weighting |
| w/o neighborhood augmentation | 71.8 | 72.5 | Remove KNN neighborhood boost |

### Key Findings

- HAMR achieves the best Macro-F1 on all 6 datasets, with the largest margin on highly imbalanced datasets (Cyclone-Idai19, IR=98.4), yielding a +1.9 pp improvement.
- Both modules contribute synergistically — removing either one degrades performance, though the meta-weighting module contributes slightly more than resampling.
- Neighborhood augmentation provides consistent marginal gains (+0.6–0.9 pp), validating the benefit of propagating difficulty from point-level to region-level.

## Highlights & Insights

- The bi-level meta-optimization "try-then-revise" strategy is elegant — pre-meta weights serve as a "draft," and feedback from the meta-validation set teaches the weight network what weight assignments truly benefit generalization. This paradigm is transferable to any scenario requiring dynamic adjustment of training priorities.
- The neighborhood-augmented resampling approach is distinctive — hard samples are treated as "seeds" from which difficulty signals are diffused through semantic neighborhoods, yielding greater robustness than attending only to isolated hard instances.
- The unified framework cleanly decouples "what to attend to" from "what to learn from" — weights govern *how* to learn, while resampling governs *what* to learn from.

## Limitations & Future Work

- Reliance on FAISS for KNN may introduce computational bottlenecks on very large datasets.
- Meta-validation set construction assumes a reasonable class distribution prior.
- Validation is limited to BERT-based encoders; applicability in the LLM era remains unexplored.
- Integration with synthetic data augmentation methods has not been investigated.

## Related Work & Insights

- **vs. Focal Loss / Dice Loss**: Static heuristics do not differentiate instance-level difficulty; HAMR dynamically learns per-instance weights.
- **vs. Meta-Weight-Net**: A similar meta-learning framework but without neighborhood resampling; HAMR additionally introduces region-level training distribution adjustment.
- **vs. SMOTE**: Generates synthetic samples rather than dynamically adjusting weights of existing samples; HAMR is more lightweight and free from generation noise.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of bi-level meta-optimization and neighborhood resampling is novel, though individual components have precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Six datasets across two tasks with detailed ablations, but comparisons with more recent methods are limited.
- **Writing Quality**: ⭐⭐⭐⭐ Method is clearly presented with a complete algorithmic pseudocode.
- **Value**: ⭐⭐⭐⭐ Provides a general and effective solution to class imbalance in NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Toward Long-Tailed Online Anomaly Detection through Class-Agnostic Concepts](../../ICCV2025/medical_imaging/toward_long-tailed_online_anomaly_detection_through_class-agnostic_concepts.md)
- [\[CVPR 2026\] Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](../../CVPR2026/medical_imaging/semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] SCDL: Semantic Class Distribution Learning for Debiasing Semi-Supervised Medical Image Segmentation](../../CVPR2026/medical_imaging/semantic_class_distribution_learning_for_debiasing.md)
- [\[CVPR 2026\] OmniFM: Toward Modality-Robust and Task-Agnostic Federated Learning for Heterogeneous Medical Imaging](../../CVPR2026/medical_imaging/omnifm_toward_modality-robust_and_task-agnostic_federated_learning_for_heterogen.md)
- [\[ACL 2026\] RADS: Reinforcement Learning-Based Sample Selection Improves Transfer Learning in Low-resource and Imbalanced Clinical Settings](rads_reinforcement_learning-based_sample_selection_improves_transfer_learning_in.md)

</div>

<!-- RELATED:END -->
